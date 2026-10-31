import asyncio
import base64
import logging
import re
import time
from pathlib import Path
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI

from core.settings import settings
from agents.construction_verification.prompts import (
    DOCUMENT_EXTRACTION_SYSTEM_PROMPT,
    DRONE_VISION_SYSTEM_PROMPT,
    SATELLITE_VISION_SYSTEM_PROMPT,
)
from PIL import Image
from agents.construction_verification.schemas import (
    CandidateClassification,
    ConstructionClaim,
    DiscoveryCandidateFinding,
    DiscoveryVisionStructuredOutput,
    DroneAnalysisOutput,
    DroneVerificationFinding,
    ExtractedProjectInfo,
    LocationInfo,
    VisionAnalysisOutput,
    VisionClaimFinding,
    VisionExecutionAudit,
    VisionModelResponse,
)

logger = logging.getLogger(__name__)

AZURE_TIMEOUT_SECONDS = 120.0


def encode_image_base64(image_path: str | Path) -> str:
    """Encode an image on disk to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _is_rate_limit(e: Exception) -> bool:
    """Determine if an exception represents a 429 / ResourceExhausted rate limit."""
    msg = str(e).lower()
    return any(
        k in msg
        for k in [
            "429",
            "resource_exhausted",
            "quota",
            "rate limit",
            "too many requests",
            "exceeded your current quota",
        ]
    )


def _is_model_not_found(e: Exception) -> bool:
    """Determine if an exception represents a 404 / NotFound / model retired error."""
    msg = str(e).lower()
    return any(
        k in msg
        for k in [
            "404",
            "not_found",
            "not found",
            "is not found",
            "no longer supported",
            "is retired",
            "model not found",
        ]
    ) and not _is_rate_limit(e)


class GeminiExtractionError(Exception):
    """Gemini could not produce structured project details from the document text."""


class GeminiVerificationClient:
    """Client for Gemini text extraction and multimodal vision analysis."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None):
        self.provider = settings.CONSTRUCTION_LLM_PROVIDER
        if self.provider == "azure":
            self._init_azure(api_key, model_name)
            return

        self.missing_config = "GEMINI_API_KEY is not configured."
        if api_key:
            self.api_key = api_key
        elif settings.GEMINI_API_KEY:
            self.api_key = settings.GEMINI_API_KEY.get_secret_value()
        elif settings.GOOGLE_API_KEY:
            self.api_key = settings.GOOGLE_API_KEY.get_secret_value()
        else:
            self.api_key = None

        self.model_name = model_name or settings.GEMINI_MODEL or "gemini-3.8-flash"
        self.vision_model = self.model_name
        self.fallback_model = settings.GEMINI_FALLBACK_MODEL or "gemini-2.5-pro"

    def _init_azure(self, api_key: str | None, model_name: str | None) -> None:
        secret = settings.CONSTRUCTION_AZURE_API_KEY
        key = api_key or (secret.get_secret_value() if secret else None)
        self.model_name = model_name or settings.CONSTRUCTION_AZURE_DEPLOYMENT
        self.vision_model = settings.CONSTRUCTION_AZURE_VISION_DEPLOYMENT or self.model_name
        self.fallback_model = None
        missing = [
            name
            for name, value in (
                ("CONSTRUCTION_AZURE_API_KEY", key),
                ("CONSTRUCTION_AZURE_ENDPOINT", settings.CONSTRUCTION_AZURE_ENDPOINT),
                ("CONSTRUCTION_AZURE_DEPLOYMENT", self.model_name),
            )
            if not value
        ]
        self.api_key = None if missing else key
        self.missing_config = f"Azure is not configured. Missing: {', '.join(missing)}."

    def _invoke_with_retry(self, llm, messages, max_retries: int = 3, initial_delay: float = 2.0):
        """Invoke LLM synchronously with exponential backoff on 429 / ResourceExhausted."""
        if settings.USE_FAKE_MODEL:
            max_retries = 0
            initial_delay = 0.0
        delay = initial_delay
        for attempt in range(max_retries + 1):
            try:
                return llm.invoke(messages)
            except Exception as e:
                if _is_rate_limit(e) and attempt < max_retries:
                    logger.warning(
                        f"Gemini API rate limit hit (attempt {attempt + 1}/{max_retries}). "
                        f"Backing off for {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    raise e

    async def _ainvoke_with_retry(self, llm, messages, max_retries: int = 3, initial_delay: float = 2.0):
        """Invoke LLM asynchronously with exponential backoff on 429 / ResourceExhausted."""
        if settings.USE_FAKE_MODEL:
            max_retries = 0
            initial_delay = 0.0
        delay = initial_delay
        for attempt in range(max_retries + 1):
            try:
                return await llm.ainvoke(messages)
            except Exception as e:
                if _is_rate_limit(e) and attempt < max_retries:
                    logger.warning(
                        f"Gemini API rate limit hit (attempt {attempt + 1}/{max_retries}). "
                        f"Backing off for {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                    delay *= 2.0
                else:
                    raise e

    def _get_llm(self, structured_schema: type | None = None, model_name: str | None = None):
        """Instantiate the configured chat model."""
        if not self.api_key or settings.USE_FAKE_MODEL:
            return None

        name = model_name or self.model_name
        if self.provider == "azure":
            # Temperature is omitted because some Azure models reject non-default values.
            # max_retries=0 leaves retries to _ainvoke_with_retry so they do not multiply.
            model = AzureChatOpenAI(
                azure_endpoint=settings.CONSTRUCTION_AZURE_ENDPOINT,
                azure_deployment=name,
                api_key=self.api_key,
                api_version=settings.CONSTRUCTION_AZURE_API_VERSION,
                timeout=AZURE_TIMEOUT_SECONDS,
                max_retries=0,
            )
            if structured_schema:
                # Strict JSON-schema mode rejects these schemas' default values.
                return model.with_structured_output(structured_schema, method="function_calling")
            return model

        model = ChatGoogleGenerativeAI(
            model=name,
            google_api_key=self.api_key,
            temperature=0.1,
            timeout=45.0,
        )
        if structured_schema:
            return model.with_structured_output(structured_schema)
        return model

    def _invoke_with_fallback(self, messages, structured_schema: type | None = None) -> tuple[Any, str]:
        """Invoke LLM synchronously. If primary model returns 404/not found, fall back to GEMINI_FALLBACK_MODEL.
        Never fall back on 429 or other failures. Returns (result, model_used).
        """
        llm = self._get_llm(structured_schema=structured_schema, model_name=self.model_name)
        if llm is None:
            raise RuntimeError("LLM not available (no API key or fake mode)")
        try:
            res = self._invoke_with_retry(llm, messages)
            return res, self.model_name
        except Exception as e:
            if _is_model_not_found(e) and self.fallback_model and self.fallback_model != self.model_name:
                logger.warning(
                    f"Gemini model {self.model_name} failed with 404/retired: {e}. "
                    f"Falling back to {self.fallback_model}..."
                )
                fallback_llm = self._get_llm(structured_schema=structured_schema, model_name=self.fallback_model)
                res = self._invoke_with_retry(fallback_llm, messages)
                return res, self.fallback_model
            raise e

    async def _ainvoke_with_fallback(
        self, messages, structured_schema: type | None = None, model_name: str | None = None
    ) -> tuple[Any, str]:
        """Invoke LLM asynchronously. If primary model returns 404/not found, fall back to GEMINI_FALLBACK_MODEL.
        Never fall back on 429 or other failures. Returns (result, model_used).
        """
        primary = model_name or self.model_name
        llm = self._get_llm(structured_schema=structured_schema, model_name=primary)
        if llm is None:
            raise RuntimeError("LLM not available (no API key or fake mode)")
        try:
            res = await self._ainvoke_with_retry(llm, messages)
            return res, primary
        except Exception as e:
            if _is_model_not_found(e) and self.fallback_model and self.fallback_model != primary:
                logger.warning(
                    f"Gemini model {self.model_name} failed with 404/retired: {e}. "
                    f"Falling back to {self.fallback_model}..."
                )
                fallback_llm = self._get_llm(structured_schema=structured_schema, model_name=self.fallback_model)
                res = await self._ainvoke_with_retry(fallback_llm, messages)
                return res, self.fallback_model
            raise e

    @staticmethod
    def _is_physical_construction_claim(text: str) -> bool:
        t = text.lower().strip()
        non_const_patterns = [
            r"period\s+\d",
            r"claim\s+period",
            r"transportation",
            r"logistics",
            r"billing\s+category",
            r"invoice\s+no",
            r"tax\s+invoice",
            r"terms\s+and\s+conditions",
            r"payment\s+voucher",
            r"bank\s+details",
        ]
        if any(re.search(pat, t) for pat in non_const_patterns):
            if not any(k in t for k in ["road", "culvert", "drain", "paving", "earthwork", "clearing", "bridge", "building", "concrete", "excavation", "macadam"]):
                return False

        physical_keywords = [
            "road", "earthwork", "clearing", "excavation", "paving", "macadam",
            "bituminous", "asphalt", "culvert", "drainage", "drain", "concrete",
            "foundation", "piling", "building", "grading", "embankment", "pier",
            "bridge", "strip", "stripping", "leveling", "sub-base", "pipeline",
            "structure", "surfacing", "widening", "link road", "formation", "site preparation",
            "subgrade", "stabilization", "runway", "pavement", "lighting", "markings", "finishing", "base layers"
        ]
        return any(k in t for k in physical_keywords)

    @classmethod
    def _filter_and_clean_claims(cls, extracted: ExtractedProjectInfo) -> None:
        valid_claims = []
        for c in extracted.claims:
            if cls._is_physical_construction_claim(c.activity):
                valid_claims.append(c)
            else:
                extracted.document_notes.append(f"Non-construction item: {c.activity}")
        extracted.claims = valid_claims

    async def extract_project_info(self, ocr_text: str) -> ExtractedProjectInfo:
        """Extract structured construction claims and project metadata from document text."""
        if settings.USE_FAKE_MODEL:
            logger.info("USE_FAKE_MODEL=true: using the rule-based test extractor instead of Gemini")
            return self._fallback_extract_claims(ocr_text)
        if not self.api_key:
            raise GeminiExtractionError(self.missing_config)

        messages = [
            SystemMessage(content=DOCUMENT_EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Please parse the following construction claim document OCR text:\n\n"
                    f"```text\n{ocr_text}\n```"
                )
            ),
        ]

        try:
            result, model_used = await self._ainvoke_with_fallback(messages, structured_schema=ExtractedProjectInfo)
            if isinstance(result, ExtractedProjectInfo):
                extracted = result
            elif isinstance(result, dict):
                extracted = ExtractedProjectInfo(**result)
            else:
                raise GeminiExtractionError(f"The model returned no structured data (got {type(result).__name__}).")

            self._filter_and_clean_claims(extracted)

            if extracted.location is None:
                extracted.location = LocationInfo()
            if extracted.location.latitude is None or extracted.location.longitude is None:
                if "site_coordinates" not in extracted.missing_fields:
                    extracted.missing_fields.append("site_coordinates")
            if extracted.start_date is None or extracted.completion_date is None:
                if "project_dates" not in extracted.missing_fields:
                    extracted.missing_fields.append("project_dates")
            return extracted
        except GeminiExtractionError:
            raise
        except Exception as e:
            logger.error(f"Gemini structured extraction failed: {e}")
            raise GeminiExtractionError(f"Structured extraction failed: {e}") from e

    async def analyze_satellite_imagery(
        self,
        claims: list[dict[str, Any]],
        before_image_path: str,
        after_image_path: str,
        metrics: dict[str, Any],
        before_date: str,
        after_date: str,
        before_scene_id: str | None = None,
        after_scene_id: str | None = None,
        raster_dimensions: tuple[int, int] | None = None,
    ) -> VisionAnalysisOutput:
        """Analyze before and after satellite images against specific construction claims with full instrumentation."""
        p_before = Path(before_image_path)
        p_after = Path(after_image_path)

        # Audit instrumentation initialization
        audit = VisionExecutionAudit(
            before_scene_id=before_scene_id or metrics.get("before_scene_id"),
            after_scene_id=after_scene_id or metrics.get("after_scene_id"),
            before_acquisition_date=before_date,
            after_acquisition_date=after_date,
            raster_dimensions=raster_dimensions or (100, 100),
            source_resolution_m=10.0,
            gemini_model_id=self.model_name,
        )

        if not p_before.exists() or not p_after.exists():
            err_msg = f"Satellite image files missing on disk: before_exists={p_before.exists()}, after_exists={p_after.exists()}"
            logger.error(err_msg)
            audit.api_call_status = "FAILED"
            audit.error_message = err_msg
            audit.parsed_response_status = "VISION_NOT_EXECUTED"
            audit.vision_actually_executed = False

            return VisionAnalysisOutput(
                findings=self._vision_not_executed(claims, err_msg, "Image artifacts missing from local storage."),
                general_observations="Multimodal vision analysis could not be executed because image files are missing.",
                vision_audit=audit,
            )

        # Record RGB image metrics and byte lengths
        try:
            with Image.open(p_before) as im1:
                audit.rgb_image_dimensions = im1.size
            audit.image_byte_lengths = [p_before.stat().st_size, p_after.stat().st_size]
            audit.image_mime_types = ["image/png", "image/png"]
            audit.both_images_included = True
        except Exception as img_err:
            logger.warning(f"Error inspecting image dimensions: {img_err}")

        if settings.USE_FAKE_MODEL:
            audit.api_call_status = "NOT_ATTEMPTED"
            audit.parsed_response_status = "TEST_SYNTHETIC"
            audit.error_message = "USE_FAKE_MODEL=true: Gemini was not called."
            out = self._fallback_satellite_vision(claims, metrics)
            out.vision_audit = audit
            return out

        if not self.api_key:
            audit.api_call_status = "NOT_ATTEMPTED"
            audit.parsed_response_status = "VISION_NOT_EXECUTED"
            audit.error_message = self.missing_config
            return VisionAnalysisOutput(
                findings=self._vision_not_executed(claims, audit.error_message, "Vision analysis did not run."),
                general_observations=audit.error_message,
                vision_audit=audit,
            )

        b64_before = encode_image_base64(before_image_path)
        b64_after = encode_image_base64(after_image_path)

        claims_text = "\n".join(
            [f"- Claim: {c.get('activity')} (Status: {c.get('claimed_status', 'completed')})" for c in claims]
        )

        user_prompt = f"""
OBSERVATION DATES:
- Pre-construction date: {before_date}
- Post-construction date: {after_date}

SPATIAL & RESOLUTION LIMITATIONS:
- Sentinel-2 optical imagery resolution is 10 meters per pixel.
- Sub-surface excavation depth, rebar, foundation stability, concrete curing, and internal structural works are INVISIBLE from space.

COMPUTED REMOTE SENSING CHANGE METRICS:
- Mean NDVI Before: {metrics.get('mean_ndvi_before')}
- Mean NDVI After: {metrics.get('mean_ndvi_after')}
- Delta NDVI: {metrics.get('delta_ndvi')}
- Vegetation Loss Fraction: {metrics.get('vegetation_loss_fraction')}
- Estimated Disturbed Area: {metrics.get('potential_disturbed_area_m2')} m^2
- Bare Soil Index (BSI) Delta: {metrics.get('delta_bsi')}
- Valid Observation Quality: {metrics.get('valid_observation_coverage_pct')}%

CLAIMS TO EVALUATE:
{claims_text}

Examine the two attached satellite images (Image 1 = Before, Image 2 = After) and the spectral change metrics.
Return structured findings for every claim according to the system instructions.
"""

        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_before}"},
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_after}"},
            },
        ]

        messages = [
            SystemMessage(content=SATELLITE_VISION_SYSTEM_PROMPT),
            HumanMessage(content=content),
        ]

        try:
            result, model_used = await self._ainvoke_with_fallback(
                messages, structured_schema=VisionModelResponse, model_name=self.vision_model
            )
            out = self._parse_vision_response(result, claims)
            audit.model_used = model_used
            audit.gemini_model_id = model_used
            audit.api_call_status = "SUCCESS"
            audit.parsed_response_status = "SUCCESS"
            audit.vision_actually_executed = True
            out.vision_audit = audit
            return out
        except Exception as e:
            is_quota = _is_rate_limit(e)
            logger.warning(f"Gemini multimodal satellite analysis failed: {e}")
            audit.api_call_status = "FAILED"
            audit.error_message = f"Model API quota exhausted (HTTP 429): {e}" if is_quota else str(e)
            audit.parsed_response_status = "QUOTA_EXHAUSTED" if is_quota else "VISION_NOT_EXECUTED"
            audit.vision_actually_executed = False
            return VisionAnalysisOutput(
                findings=self._vision_not_executed(
                    claims,
                    "Model API rate limit reached (HTTP 429)." if is_quota else f"API call failed ({e}).",
                    "Vision analysis failed; physical inspection required.",
                ),
                general_observations=audit.error_message,
                vision_audit=audit,
            )

    @staticmethod
    def _vision_not_executed(
        claims: list[dict[str, Any]], reason: str, limitation: str
    ) -> list[VisionClaimFinding]:
        return [
            VisionClaimFinding(
                claim=c.get("activity", ""),
                supports_claim="VISION_NOT_EXECUTED",
                status="VISION_NOT_EXECUTED",
                reasoning=f"VISION_NOT_EXECUTED: {reason}",
                limitations=[limitation],
                recommended_follow_up="Re-run the analysis or arrange a physical inspection.",
            )
            for c in claims
        ]

    @staticmethod
    def _parse_vision_response(result: Any, claims: list[dict[str, Any]]) -> VisionAnalysisOutput:
        """Convert the model response, rejecting anything that is not a usable assessment."""
        if isinstance(result, dict):
            result = VisionModelResponse.model_validate(result)
        if not isinstance(result, VisionModelResponse):
            raise ValueError(f"The model returned no structured data (got {type(result).__name__}).")
        if claims and not result.findings:
            raise ValueError("The model returned no findings for the submitted claims.")
        supports = {"SUPPORTED": True, "NOT_SUPPORTED": False, "INSUFFICIENT_EVIDENCE": None}
        return VisionAnalysisOutput(
            findings=[
                VisionClaimFinding(
                    claim=f.claim,
                    observed_changes=f.visual_observations,
                    supports_claim=supports[f.status],
                    status=f.status,
                    reasoning=f.reasoning,
                    limitations=f.limitations,
                    recommended_follow_up=f.recommended_follow_up,
                )
                for f in result.findings
            ],
            general_observations=result.general_observations,
        )

    def analyze_discovery_imagery(
        self,
        rgb_before_path: str,
        rgb_after_path: str,
        candidates: list[dict[str, Any]],
        metrics: dict[str, Any],
        scene_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform multimodal Gemini Vision analysis for independent construction discovery."""
        p_before = Path(rgb_before_path)
        p_after = Path(rgb_after_path)

        audit = VisionExecutionAudit(
            before_scene_id=scene_metadata.get("before_scene_id"),
            after_scene_id=scene_metadata.get("after_scene_id"),
            before_acquisition_date=scene_metadata.get("before_acquisition_date"),
            after_acquisition_date=scene_metadata.get("after_acquisition_date"),
            raster_dimensions=tuple(scene_metadata.get("raster_shape", (500, 500))),
            source_resolution_m=10.0,
            gemini_model_id=self.model_name,
        )

        if not p_before.exists() or not p_after.exists():
            err_msg = "Discovery image files missing on disk"
            audit.api_call_status = "FAILED"
            audit.error_message = err_msg
            audit.parsed_response_status = "VISION_NOT_EXECUTED"
            audit.vision_actually_executed = False
            return {
                "summary": f"VISION_NOT_EXECUTED: {err_msg}",
                "candidate_interpretations": {},
                "audit": audit,
            }

        try:
            with Image.open(p_before) as im1:
                audit.rgb_image_dimensions = im1.size
            audit.image_byte_lengths = [p_before.stat().st_size, p_after.stat().st_size]
            audit.image_mime_types = ["image/png", "image/png"]
            audit.both_images_included = True
        except Exception as e:
            logger.warning(f"Failed measuring image size: {e}")

        llm = self._get_llm(DiscoveryVisionStructuredOutput)
        if llm is None:
            audit.api_call_status = "NOT_ATTEMPTED"
            audit.parsed_response_status = "TEST_SYNTHETIC" if settings.USE_FAKE_MODEL else "VISION_NOT_EXECUTED"
            audit.vision_actually_executed = False
            audit.error_message = "No Gemini API key or USE_FAKE_MODEL=True"
            fallback_findings = [
                DiscoveryCandidateFinding(
                    candidate_id=c["candidate_id"],
                    classification=CandidateClassification.CONSTRUCTION,
                    confidence=0.85 if settings.USE_FAKE_MODEL else 0.5,
                    visual_evidence=f"Detected ground disturbance and surface alteration ({c['area_hectares']} ha).",
                    limitations=["Automated spectral detection; verification required."],
                )
                for c in candidates
            ]
            interps = {
                f.candidate_id: f"{f.classification.value}: {f.visual_evidence}"
                for f in fallback_findings
            }
            summary_txt = (
                f"Independent change detection identified {len(candidates)} candidate zones "
                f"covering {metrics.get('total_disturbed_area_m2', 0):,.0f} m^2 based on multi-temporal spectral analysis."
            )
            return {
                "summary": summary_txt,
                "executive_summary": summary_txt,
                "findings": fallback_findings,
                "candidate_interpretations": interps,
                "audit": audit,
            }

        # Format candidates text for prompt
        cand_descriptions = []
        for c in candidates[:8]:
            cand_descriptions.append(
                f"- {c['candidate_id']}: Centroid ({c['centroid']['latitude']}, {c['centroid']['longitude']}), "
                f"Area: {c['area_hectares']} ha ({c['area_m2']:,.0f} m^2), Delta NDVI: {c.get('mean_delta_ndvi', 'N/A')}, "
                f"Delta BSI: {c.get('mean_delta_bsi', 'N/A')}"
            )
        cand_text = "\n".join(cand_descriptions) if cand_descriptions else "No large candidate clusters identified."

        b64_before = encode_image_base64(rgb_before_path)
        b64_after = encode_image_base64(rgb_after_path)

        prompt = f"""You are an expert remote sensing satellite imagery analyst for government infrastructure audit.
Compare Image 1 (Before: {scene_metadata.get('before_acquisition_date')}) and Image 2 (After: {scene_metadata.get('after_acquisition_date')}) for {scene_metadata.get('area_name')}.

OBSERVED CHANGE METRICS:
- Total Disturbed Area: {metrics.get('total_disturbed_area_m2', 0):,.0f} m^2
- Mean Delta NDVI: {metrics.get('delta_ndvi_mean', 0.0)}

GEOGRAPHIC CANDIDATE CHANGE ZONES (Coordinates derived from raster):
{cand_text}

CLASSIFICATION TAXONOMY:
- CONSTRUCTION: Linear paved surfaces, earthworks, foundations, building construction, runways, roads.
- AGRICULTURAL_HARVEST: Crop harvesting, seasonal drying, fallow field clearing, plowing patterns.
- WATER_BODY: Reservoir filling/drying, flood inundation, retention pond expansion.
- CLOUD_OR_ARTIFACT: Unmasked cloud edges, haze, sensor striping or terrain shadows.
- UNCLEAR: Ambiguous visual change at 10m Sentinel-2 resolution.

INSTRUCTIONS:
1. For each listed candidate ID (e.g. CANDIDATE-01, CANDIDATE-02), output a DiscoveryCandidateFinding with exact matching candidate_id, classification, confidence (0.0 to 1.0), visual_evidence, and limitations.
2. Provide an overall_summary summarizing all observed physical developments.
Be factual, objective, and describe actual visual pixel differences. Do not invent candidate IDs."""

        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_before}"},
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_after}"},
            },
        ]

        messages = [
            SystemMessage(content="You are an expert geospatial remote sensing analyst. Analyze real satellite image pairs objectively using structured output."),
            HumanMessage(content=content),
        ]

        try:
            resp, model_used = self._invoke_with_fallback(messages, structured_schema=DiscoveryVisionStructuredOutput)
            audit.model_used = model_used
            audit.gemini_model_id = model_used
            audit.api_call_status = "SUCCESS"
            audit.parsed_response_status = "SUCCESS"
            audit.vision_actually_executed = True

            if isinstance(resp, DiscoveryVisionStructuredOutput):
                structured = resp
            elif isinstance(resp, dict):
                structured = DiscoveryVisionStructuredOutput(**resp)
            else:
                structured = DiscoveryVisionStructuredOutput(candidate_findings=[], overall_summary=str(resp))

            # Validate returned candidate IDs against input candidates
            valid_ids = {c["candidate_id"] for c in candidates}
            validated_findings: list[DiscoveryCandidateFinding] = []
            for f in structured.candidate_findings:
                if f.candidate_id in valid_ids:
                    validated_findings.append(f)
                else:
                    logger.warning(f"Discarding hallucinated/unknown candidate ID from model: {f.candidate_id}")

            # Ensure any candidate not returned gets an explicit UNCLEAR entry
            returned_ids = {f.candidate_id for f in validated_findings}
            for c in candidates:
                c_id = c["candidate_id"]
                if c_id not in returned_ids:
                    validated_findings.append(
                        DiscoveryCandidateFinding(
                            candidate_id=c_id,
                            classification=CandidateClassification.UNCLEAR,
                            confidence=0.5,
                            visual_evidence=f"Candidate area ({c['area_hectares']} ha) not uniquely resolved in structured findings.",
                            limitations="Resolution or model coverage limitation.",
                        )
                    )

            cand_interps = {
                f.candidate_id: f"{f.classification.value}: {f.visual_evidence}"
                for f in validated_findings
            }

            return {
                "summary": structured.overall_summary,
                "overall_summary": structured.overall_summary,
                "findings": validated_findings,
                "candidate_interpretations": cand_interps,
                "audit": audit,
            }
        except Exception as e:
            is_quota = _is_rate_limit(e)
            err_msg = f"Gemini discovery vision call failed: {e}"
            logger.error(err_msg)
            audit.api_call_status = "FAILED"
            audit.error_message = f"Gemini API Quota Exhausted (HTTP 429): {e}" if is_quota else str(e)
            audit.parsed_response_status = "VISION_NOT_EXECUTED"
            audit.vision_actually_executed = False
            return {
                "summary": f"VISION_NOT_EXECUTED: {audit.error_message}",
                "overall_summary": f"VISION_NOT_EXECUTED: {audit.error_message}",
                "findings": [],
                "candidate_interpretations": {},
                "audit": audit,
            }


    async def analyze_drone_imagery(
        self,
        claims: list[dict[str, Any]],
        drone_image_paths: list[str],
    ) -> list[DroneVerificationFinding]:
        """Analyze high-resolution drone or on-site photographs against unresolved claims using structured multimodal vision."""
        if not drone_image_paths:
            return [
                DroneVerificationFinding(
                    claim=c.get("activity", "Unknown"),
                    observations=[],
                    supports_claim="VISION_NOT_EXECUTED",
                    reasoning="VISION_NOT_EXECUTED: No on-site or drone inspection photographs were provided.",
                    limitations=["Missing inspection imagery."],
                )
                for c in claims
            ]

        valid_images = [p for p in drone_image_paths if Path(p).exists()]
        if not valid_images:
            return [
                DroneVerificationFinding(
                    claim=c.get("activity", "Unknown"),
                    observations=[],
                    supports_claim="VISION_NOT_EXECUTED",
                    reasoning="VISION_NOT_EXECUTED: Specified inspection image files were not found on disk.",
                    limitations=["Missing image files."],
                )
                for c in claims
            ]

        if not self.api_key and not settings.USE_FAKE_MODEL:
            return [
                DroneVerificationFinding(
                    claim=c.get("activity", "Unknown"),
                    observations=[],
                    supports_claim="VISION_NOT_EXECUTED",
                    reasoning="VISION_NOT_EXECUTED: GEMINI_API_KEY is not configured.",
                    limitations=["API credentials missing."],
                )
                for c in claims
            ]

        if settings.USE_FAKE_MODEL:
            return self._fallback_drone_eval(claims, drone_image_paths)

        # Build prompt and image parts
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are evaluating high-resolution on-site inspection photographs against the following claims:\n"
                    + "\n".join([f"- Claim: {c.get('activity')}" for c in claims])
                    + "\n\nFor each claim, determine whether the physical photograph supports the claim, contradicts it, or provides insufficient evidence. Never invent observations."
                ),
            }
        ]

        for p in valid_images[:4]:  # limit to 4 images
            b64_img = encode_image_base64(p)
            mime = "image/png" if Path(p).suffix.lower() == ".png" else "image/jpeg"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64_img}"},
            })

        messages = [
            SystemMessage(content=DRONE_VISION_SYSTEM_PROMPT),
            HumanMessage(content=content),
        ]

        try:
            llm = self._get_llm(DroneAnalysisOutput)
            if llm is None:
                return [
                    DroneVerificationFinding(
                        claim=c.get("activity", "Unknown"),
                        observations=[],
                        supports_claim="VISION_NOT_EXECUTED",
                        reasoning="VISION_NOT_EXECUTED: Model could not be instantiated.",
                        limitations=["Model initialization failed."],
                    )
                    for c in claims
                ]

            resp = await self._ainvoke_with_retry(llm, messages)
            if isinstance(resp, DroneAnalysisOutput):
                parsed_findings = resp.findings
            elif isinstance(resp, dict):
                parsed_findings = DroneAnalysisOutput(**resp).findings
            else:
                parsed_findings = []

            # Ensure every claim has a corresponding finding
            claim_map = {f.claim.strip().lower(): f for f in parsed_findings}
            final_findings: list[DroneVerificationFinding] = []
            for c in claims:
                act = c.get("activity", "Unknown").strip()
                match = claim_map.get(act.lower())
                if match:
                    final_findings.append(match)
                else:
                    final_findings.append(
                        DroneVerificationFinding(
                            claim=act,
                            observations=["Claim not specifically resolved in photographic analysis."],
                            supports_claim="INSUFFICIENT_EVIDENCE",
                            reasoning="Inspection photograph did not contain verifiable evidence for this specific activity.",
                            limitations=["Occluded or omitted from camera angle."],
                        )
                    )
            return final_findings

        except Exception as e:
            is_quota = _is_rate_limit(e)
            logger.error(f"Drone vision analysis failed: {e}")
            tag = "QUOTA_EXHAUSTED" if is_quota else "VISION_NOT_EXECUTED"
            return [
                DroneVerificationFinding(
                    claim=c.get("activity", "Unknown"),
                    observations=[],
                    supports_claim=tag,
                    reasoning=f"{tag}: Inspection vision model failed ({e}).",
                    limitations=["Vision API rate limit or execution error."],
                )
                for c in claims
            ]

    def _fallback_extract_claims(self, text: str) -> ExtractedProjectInfo:
        """Deterministic heuristic extraction from text."""
        import re
        from datetime import datetime

        project_name = "Infrastructure Project"
        construction_type = "Civil Construction"
        start_date = None
        completion_date = None
        latitude = None
        longitude = None
        claims: list[ConstructionClaim] = []
        missing_fields: list[str] = []

        # Project name
        for line in text.splitlines():
            line_s = line.strip()
            m = re.search(r"project(?:\s+name)?[:\t\s]+([^\t\n\r]+)", line_s, re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                if len(cand) > 3 and not cand.lower().startswith("details") and not cand.lower().startswith("developer"):
                    project_name = cand
                    break
            elif "project:" in line_s.lower():
                project_name = line_s.split(":", 1)[1].strip()

        # Construction type
        for line in text.splitlines():
            line_s = line.strip()
            if "construction type:" in line_s.lower():
                construction_type = line_s.split(":", 1)[1].strip()
        if construction_type == "Civil Construction":
            if "runway" in text.lower() or "airport" in text.lower():
                construction_type = "Runway Construction"
            elif "highway" in text.lower():
                construction_type = "Highway Construction"

        # Coordinates regex
        # 1. Midpoint coordinates with degrees / cardinal directions
        mid_m = re.search(r"midpoint[^\d]*(\d{1,2}\.\d+)\s*°?\s*N[^\d]*(\d{1,3}\.\d+)\s*°?\s*E", text, re.IGNORECASE)
        if mid_m:
            try:
                latitude = float(mid_m.group(1))
                longitude = float(mid_m.group(2))
            except ValueError:
                pass

        # 2. General degree coordinates: 28.183467° N, 77.592436° E
        if latitude is None or longitude is None:
            deg_m = re.search(r"(\d{1,2}\.\d+)\s*°?\s*N[^\d]*(\d{1,3}\.\d+)\s*°?\s*E", text, re.IGNORECASE)
            if deg_m:
                try:
                    latitude = float(deg_m.group(1))
                    longitude = float(deg_m.group(2))
                except ValueError:
                    pass

        # 3. Labeled latitude / longitude
        if latitude is None or longitude is None:
            lat_match = re.search(r"latitude[:\s\t]+([-+]?[0-9]*\.?[0-9]+)", text, re.IGNORECASE)
            lon_match = re.search(r"longitude[:\s\t]+([-+]?[0-9]*\.?[0-9]+)", text, re.IGNORECASE)
            if lat_match and lon_match:
                try:
                    latitude = float(lat_match.group(1))
                    longitude = float(lon_match.group(1))
                except ValueError:
                    pass

        # 4. Decimal pair: 12.3456, 78.9012
        if latitude is None or longitude is None:
            pair_match = re.search(r"([-+]?\d{1,2}\.\d+)[,\s]+([-+]?\d{1,3}\.\d+)", text)
            if pair_match:
                try:
                    latitude = float(pair_match.group(1))
                    longitude = float(pair_match.group(2))
                except ValueError:
                    pass

        # Dates regex
        # 1. Explicit labeled Start Date / Completion Date (YYYY-MM-DD or YYYY/MM/DD)
        start_lbl = re.search(r"start\s+date[:\s\t]+([0-9\-/]+)", text, re.IGNORECASE)
        if start_lbl:
            start_date = start_lbl.group(1).strip().replace("/", "-")
        comp_lbl = re.search(r"completion\s+date[:\s\t]+([0-9\-/]+)", text, re.IGNORECASE)
        if comp_lbl:
            completion_date = comp_lbl.group(1).strip().replace("/", "-")

        # 2. Labeled commencement / completion with Month name (e.g. 25 November 2021)
        months_pat = r"January|February|March|April|May|June|July|August|September|October|November|December"
        if start_date is None:
            comm_m = re.search(rf"(?:commencement|start|launch|groundbreaking)[^\w\n]*(\d{{1,2}}\s+(?:{months_pat})\s+20\d\d)", text, re.IGNORECASE)
            if comm_m:
                try:
                    start_date = datetime.strptime(comm_m.group(1).strip(), "%d %B %Y").strftime("%Y-%m-%d")
                except Exception:
                    pass

        if completion_date is None:
            comp_m = re.search(rf"(?:completion|completed|finish)[^\w\n]*(\d{{1,2}}\s+(?:{months_pat})\s+20\d\d)", text, re.IGNORECASE)
            if comp_m:
                try:
                    completion_date = datetime.strptime(comp_m.group(1).strip(), "%d %B %Y").strftime("%Y-%m-%d")
                except Exception:
                    pass

        # 2. Date ranges in Month name: DD Month YYYY - DD Month YYYY
        if start_date is None or completion_date is None:
            range_m = re.search(rf"(\d{{1,2}}\s+(?:{months_pat})\s+20\d\d)\s*[-–]\s*(\d{{1,2}}\s+(?:{months_pat})\s+20\d\d)", text, re.IGNORECASE)
            if range_m:
                try:
                    if start_date is None:
                        start_date = datetime.strptime(range_m.group(1).strip(), "%d %B %Y").strftime("%Y-%m-%d")
                    if completion_date is None:
                        completion_date = datetime.strptime(range_m.group(2).strip(), "%d %B %Y").strftime("%Y-%m-%d")
                except Exception:
                    pass

        # 3. Standard YYYY-MM-DD
        if start_date is None or completion_date is None:
            dates = re.findall(r"\b(20\d\d[-/]\d\d[-/]\d\d)\b", text)
            if len(dates) >= 2:
                if start_date is None:
                    start_date = dates[0].replace("/", "-")
                if completion_date is None:
                    completion_date = dates[1].replace("/", "-")
            elif len(dates) == 1 and start_date is None:
                start_date = dates[0].replace("/", "-")

        # Claims extraction
        document_notes: list[str] = []

        # 1. Itemized table lines (e.g. S.No Description Amount)
        def clean_table_activity(line: str) -> str:
            l = re.sub(r'^\s*\d+[\.\t\s]*', '', line).strip()
            l = re.sub(r'[\t\s]+[\*₹マ〈]*\s*[\d,]+.*$', '', l).strip()
            return l

        ignore_claim_words = [
            'expenditure', 'particulars', 'declaration', 'summary', 'details', 'threshold',
            'dimensions', 'identification', 'surface', 'midpoint', 'completion', 'amounts claimed',
            'subject to', 'reimbursement', 'statement', 'supported by', 'description', 'amount',
            'machinery', 'labour', 'personnel', 'testing', 'transportation', 'logistics', 'other approved',
            'claimant', 'developer', 'contractor', 'company', 'signatory', 'subcontractor',
            'noida', 'yamuna', 'international', 'reimbursement claim'
        ]

        for line in text.splitlines():
            l_clean = clean_table_activity(line)
            l_low = l_clean.lower()
            if 5 < len(l_clean) < 80 and self._is_physical_construction_claim(l_clean):
                if not any(ign in l_low for ign in ignore_claim_words):
                    if not any(l_clean.lower() in c.activity.lower() or c.activity.lower() in l_clean.lower() for c in claims):
                        claims.append(ConstructionClaim(activity=l_clean, claimed_status="completed"))

        # 2. Explicit "claim X: ..." patterns
        explicit_claims = re.findall(r"claim\s*\d*[:\s]+([^.\n]+)", text, re.IGNORECASE)
        if explicit_claims:
            for m in explicit_claims:
                m_str = m.strip()
                if len(m_str) > 3:
                    if self._is_physical_construction_claim(m_str):
                        if not any(m_str.lower() in c.activity.lower() or c.activity.lower() in m_str.lower() for c in claims):
                            claims.append(ConstructionClaim(activity=m_str, claimed_status="completed"))
                    else:
                        document_notes.append(f"Non-construction note: {m_str}")

        # 3. Keyword matches if still empty
        if not claims:
            keyword_matches = re.findall(
                r"(?:vegetation clearing|clearing|excavation|earthwork|grading|paving|culvert|foundation|bituminous|drainage)[^.\n]*",
                text,
                re.IGNORECASE,
            )
            for m in keyword_matches:
                m_str = m.strip()
                if len(m_str) > 5 and self._is_physical_construction_claim(m_str):
                    if not any(m_str.lower() in c.activity.lower() or c.activity.lower() in m_str.lower() for c in claims):
                        claims.append(ConstructionClaim(activity=m_str, claimed_status="completed"))
                elif len(m_str) > 5:
                    document_notes.append(f"Non-construction note: {m_str}")

        # 4. Scan lines for non-construction items (dates, billing categories, transportation/logistics)
        for line in text.splitlines():
            l_str = line.strip()
            if any(k in l_str.lower() for k in ["transportation", "logistics", "period 01", "billing category", "invoice", "personnel", "machinery", "labour"]) and not self._is_physical_construction_claim(l_str):
                if l_str not in document_notes:
                    document_notes.append(l_str)

        # 5. Default fallback claims if still empty
        if not claims:
            # Add general default claims based on text content only if real physical keywords exist
            if "clearing" in text.lower() or "vegetation" in text.lower():
                claims.append(ConstructionClaim(activity="Vegetation clearing and site preparation", claimed_status="completed"))
            elif "excavation" in text.lower() or "foundation" in text.lower():
                claims.append(ConstructionClaim(activity="Foundation excavation", claimed_status="completed"))
            elif "runway" in text.lower():
                claims.append(ConstructionClaim(activity="Runway pavement construction (asphalt)", claimed_status="completed"))
            elif "road" in text.lower() or "paving" in text.lower() or "highway" in text.lower():
                claims.append(ConstructionClaim(activity="Road construction and surfacing", claimed_status="completed"))

        if latitude is None or longitude is None:
            missing_fields.append("site_coordinates")
        if start_date is None or completion_date is None:
            missing_fields.append("project_dates")

        # Build bounding box if coordinates exist (+- 0.005 degrees ~ 500m)
        bbox = None
        if latitude is not None and longitude is not None:
            bbox = [
                round(longitude - 0.005, 6),
                round(latitude - 0.005, 6),
                round(longitude + 0.005, 6),
                round(latitude + 0.005, 6),
            ]

        location_info = LocationInfo(
            site_name=project_name,
            latitude=latitude,
            longitude=longitude,
            bbox=bbox,
        )

        return ExtractedProjectInfo(
            project_name=project_name,
            construction_type=construction_type,
            location=location_info,
            start_date=start_date,
            completion_date=completion_date,
            claims=claims,
            document_notes=document_notes,
            missing_fields=missing_fields,
            uncertain_fields=[],
        )

    def _fallback_satellite_vision(
        self, claims: list[dict[str, Any]], metrics: dict[str, Any]
    ) -> VisionAnalysisOutput:
        """Deterministic fallback for satellite vision assessment based on strict physical constraints."""
        findings: list[VisionClaimFinding] = []
        loss_fraction = metrics.get("vegetation_loss_fraction", 0.0)
        disturbed_area = metrics.get("potential_disturbed_area_m2", 0.0)

        for c in claims:
            activity = c.get("activity", "").lower()

            if any(kw in activity for kw in ["clearing", "vegetation", "site prep", "brush"]):
                # Surface vegetation clearing can be directly supported by significant NDVI reduction
                if loss_fraction >= 0.15:
                    findings.append(
                        VisionClaimFinding(
                            claim=c.get("activity", ""),
                            observed_changes=[
                                f"Widespread surface vegetation loss detected ({loss_fraction * 100:.1f}% of site)",
                                f"Spectral change indicates ~{disturbed_area:.0f} m^2 of disturbed ground",
                            ],
                            supports_claim=True,
                            reasoning="Significant reduction in NDVI and exposed soil signature confirms surface vegetation clearing.",
                            limitations=["Sentinel-2 10m resolution does not confirm fine perimeter grading or biomass disposal."],
                        )
                    )
                else:
                    findings.append(
                        VisionClaimFinding(
                            claim=c.get("activity", ""),
                            observed_changes=["Vegetation index shows minimal change across target area."],
                            supports_claim=False,
                            reasoning="Satellite spectral indices show insufficient vegetation reduction to support clearing claim.",
                            limitations=["Cloud shadows or seasonal regrowth may obscure partial clearing."],
                        )
                    )
            elif any(kw in activity for kw in ["foundation", "excavation", "trench", "culvert", "structural", "concrete"]):
                # Sub-surface or structural claims CANNOT be confirmed by 10m Sentinel-2 satellite imagery!
                findings.append(
                    VisionClaimFinding(
                        claim=c.get("activity", ""),
                        observed_changes=[
                            "Ground disturbance and exposed soil are visible on site.",
                            "However, trench depth, foundation dimensions, rebar, and structural completion cannot be resolved at 10m spatial resolution.",
                        ],
                        supports_claim=None,
                        reasoning="Optical satellite imagery cannot confirm sub-surface excavation depth, structural reinforcement, or foundation engineering quality.",
                        limitations=[
                            "Sentinel-2 10m resolution cannot resolve subterranean structures or concrete reinforcement",
                            "Ground appearance change does not prove engineering completion",
                        ],
                    )
                )
            else:
                # Default for other activities
                findings.append(
                    VisionClaimFinding(
                        claim=c.get("activity", ""),
                        observed_changes=["General surface spectral alteration detected on site."],
                        supports_claim=None,
                        reasoning="Satellite observations are inconclusive for this specific construction activity.",
                        limitations=["10m optical satellite resolution is insufficient for fine-scale construction validation."],
                    )
                )

        return VisionAnalysisOutput(
            findings=findings,
            general_observations="Satellite multi-temporal comparison shows marked ground surface changes consistent with site activation.",
        )

    def _fallback_drone_eval(
        self, claims: list[dict[str, Any]], drone_image_paths: list[str]
    ) -> list[DroneVerificationFinding]:
        """Deterministic fallback for test suites and offline mock mode."""
        findings: list[DroneVerificationFinding] = []
        for c in claims:
            act = c.get("activity", "")
            is_culvert = any(kw in act.lower() for kw in ["culvert", "drain", "pipe"])
            findings.append(
                DroneVerificationFinding(
                    claim=act,
                    observations=[
                        f"Photographic inspection evidence evaluated ({len(drone_image_paths)} images).",
                        f"Physical surface indicators of {act} checked.",
                    ],
                    supports_claim="INSUFFICIENT_EVIDENCE" if is_culvert else "SUPPORTED",
                    reasoning=f"Photographic verification evaluation completed for {act}.",
                    limitations=[
                        "Optical photograph verifies surface appearance only.",
                        "Material strength and internal structural tests require physical laboratory certificates.",
                    ],
                )
            )
        return findings
