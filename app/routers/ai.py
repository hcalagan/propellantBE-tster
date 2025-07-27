from fastapi import APIRouter, HTTPException, Depends
import httpx
import os
import logging
from app.models.requests import CVAnalysisRequest, CVAnalysis
from app.services.input_service import InputService
from app.services.ai_service import AIService


# Configure logging
logger = logging.getLogger(__name__)

# Determine if we're in development mode
DEV_MODE = os.getenv("ENVIRONMENT", "development") == "development"

# Initialize FastAPI router
router = APIRouter()

@router.post("/cv-analysis")
async def cv_analysis(
    cv_data: CVAnalysisRequest
):
    try:
        # Validate input
        validated_data = InputService.validate_input(cv_data)

        # Process the request directly
        enhanced_cv = await AIService.rewrite_content(validated_data)

        # Check if it's already a dict or needs conversion
        if hasattr(enhanced_cv, 'model_dump'):
            return enhanced_cv.model_dump()
        else:
            # It's already a dict, return directly
            return enhanced_cv

    except Exception as e:
        # Add general exception handling
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")