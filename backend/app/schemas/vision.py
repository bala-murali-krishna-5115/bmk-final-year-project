from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class VisionQualitySchema(BaseModel):
    brightness_mean: Optional[float] = None
    brightness_score: Optional[float] = None
    blur_variance: Optional[float] = None
    blur_score: Optional[float] = None
    face_size_score: Optional[float] = None
    stability_score: Optional[float] = None
    overall_quality: float = Field(..., ge=0.0, le=100.0)
    status: Optional[str] = "good"

class VisionFaceSchema(BaseModel):
    face_detected: bool
    face_count: int = Field(..., ge=0)
    bbox: Optional[Dict[str, int]] = None

class VisionEyesSchema(BaseModel):
    left_eye_openness: float = Field(..., ge=0.0, le=1.0)
    right_eye_openness: float = Field(..., ge=0.0, le=1.0)
    mean_openness: float = Field(..., ge=0.0, le=1.0)
    eye_asymmetry: float = Field(..., ge=0.0)
    eye_state: Optional[str] = "open"

class VisionSymmetrySchema(BaseModel):
    facial_symmetry_score: float = Field(..., ge=0.0, le=1.0)
    left_right_difference: Optional[float] = None
    valid_for_symmetry: Optional[bool] = True

class VisionSkinSchema(BaseModel):
    skin_r_mean: float
    skin_g_mean: float
    skin_b_mean: float
    skin_h_mean: float
    skin_s_mean: float
    skin_v_mean: float
    skin_l_mean: float
    skin_a_mean: float
    skin_b_lab_mean: float
    skin_texture_lbp: float

class VisionFeaturesRequest(BaseModel):
    schema_version: str = "1.0"
    quality: VisionQualitySchema
    face: VisionFaceSchema
    eyes: VisionEyesSchema
    symmetry: VisionSymmetrySchema
    skin: VisionSkinSchema

class VisionFeaturesResponse(BaseModel):
    session_id: str
    feature_schema_version: str
    camera_quality_score: float
    status: str
    message: str
