import json
import os
import logging
from typing import Dict, List, Any
from datetime import datetime
from openai import AsyncOpenAI
from dotenv import load_dotenv
from app.models.requests import CVAnalysisRequest, CVAnalysis, Skill, Experiences

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load environment variables
load_dotenv()

# API credentials
UNIFY_URL = os.getenv("UNIFY_URL")
UNIFY_API_KEY = os.getenv("UNIFY_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

# Initialize client
client = AsyncOpenAI(
    base_url=UNIFY_URL,
    api_key=UNIFY_API_KEY
)

class AIService:
    @staticmethod
    def edit_cv():
        return {
            "type": "function",
            "function": {
                "name": "provide_edited_cv",
                "description": (
                    "Edit the CV to better match the job description. "
                    "You MUST return a new JSON object with the EXACT SAME structure as the input, including ALL original fields: 'experiences', 'skills', 'professionalSummary', and 'jobDescription'. "
                    "Do not omit or rename any fields, even if unchanged. "
                    "Enhance the content of each field as appropriate, but always include every required field in the output. "
                    "The professionalSummary must be a strong, metric-driven summary."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "experiences": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "List of optimized work experiences. This field is REQUIRED."
                        },
                        "skills": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "List of optimized skills. This field is REQUIRED."
                        },
                        "professionalSummary": {
                            "type": "string",
                            "description": "A strong, metric-driven professional summary. This field is REQUIRED."
                        },
                        "jobDescription": {
                            "type": "string",
                            "description": "Job description used for optimization. This field is REQUIRED."
                        }
                    },
                    "required": ["experiences", "skills", "professionalSummary", "jobDescription"]
                }
            }
        }

    @classmethod
    async def rewrite_content(cls, req: CVAnalysisRequest, rules: dict = None) -> dict:
        """Ultra-simplified approach that hard-codes what we need"""
        current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Current Date and Time: {current_time}")
        logger.info(f"User: praiseunite")
        
        try:
            # Compose prompt for function calling
            prompt = (
                f"You are a Certified Professional Resume Writer with 20+ years of experience. "
                f"Given this job description: {req.jobDescription} and the following CV data (skills and experiences), rewrite the CV to better match the job description. "
                f"You MUST return the result as a JSON object with the EXACT SAME structure as the input, including ALL original fields: 'experiences', 'skills', 'professionalSummary', and 'jobDescription'. "
                f"For each experience, rewrite the 'description field' and the 'achievements fields' using keywords and measurable goals from the job description, but DO NOT invent or hallucinate new experiences, skills, or achievements. "
                f"Keep the core meaning and facts true to the original, just express them in a way that matches the job requirements. "
                f"Do NOT omit, rename, or remove any fields, even if unchanged. Only enhance the content. "
                f"The professionalSummary must be a compelling, measurable, and concise summary that highlights the candidate's impact, skills, and achievements, using numbers, percentages, or other metrics where possible. "
                f"Base the summary on the user's work experiences and skills."
                f"\n\nCV DATA:\n{json.dumps(req.model_dump())}"
            )

            logger.info(f"Sending request to AI with function calling...")
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                tools=[cls.edit_cv()],
                tool_choice="auto"
            )

            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                logger.error(f"No tool calls received from AI.")
                return cls._fallback_response(req)


            function_args = json.loads(tool_calls[0].function.arguments)
            logger.info(f"Received tool call arguments: {function_args}")

            # Merge missing fields from the input
            if "experiences" not in function_args:
                function_args["experiences"] = [exp.model_dump() for exp in req.experiences]
            if "skills" not in function_args:
                function_args["skills"] = [skill.model_dump() for skill in req.skills]
            if "jobDescription" not in function_args:
                function_args["jobDescription"] = req.jobDescription

            # Use the CVAnalysis Pydantic model to validate and structure the response
            try:
                function_data_param = CVAnalysis.parse_obj(function_args)
            except Exception as e:
                logger.error(f"Pydantic validation error: {e}")
                return cls._fallback_response(req)

            logger.info(f"Successfully created enhanced CV with all experiences")
            return function_data_param.dict()

        except Exception as e:
            logger.error(f"Error processing CV: {str(e)}")
            return cls._fallback_response(req)
    
    @staticmethod
    def _fallback_response(req: CVAnalysisRequest) -> dict:
        """Simple fallback response"""
        return {
            "experiences": [exp.model_dump() for exp in req.experiences],
            "skills": [skill.model_dump() for skill in req.skills],
            "professionalSummary": "Experienced professional with skills matching the job requirements."
        }