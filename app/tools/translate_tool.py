"""
Translation Tool - Translate text using LLM
"""

import logging
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class TranslateTool:
    """Tool for translating text between languages"""
    
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'pt': 'Portuguese',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
    }
    
    def __init__(self):
        """Initialize translation tool"""
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        
        if self.gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_key)
                # Use gemini-2.5-flash (works with free tier)
                self.client = genai.GenerativeModel('models/gemini-2.5-flash')
                logger.info("Translation tool initialized with Gemini")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini for translation: {e}")
    
    def translate(
        self,
        text: str,
        source_lang: str = 'auto',
        target_lang: str = 'en'
    ) -> str:
        """
        Translate text from source to target language
        
        Args:
            text: Text to translate
            source_lang: Source language code (or 'auto' for detection)
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        if not self.client:
            logger.warning("Translation client not initialized")
            return text
        
        if target_lang not in self.SUPPORTED_LANGUAGES:
            logger.error(f"Unsupported target language: {target_lang}")
            return text
        
        try:
            target_lang_name = self.SUPPORTED_LANGUAGES[target_lang]
            
            if source_lang == 'auto':
                prompt = f"Translate the following text to {target_lang_name}. Only return the translation, nothing else:\n\n{text}"
            else:
                source_lang_name = self.SUPPORTED_LANGUAGES.get(source_lang, source_lang)
                prompt = f"Translate from {source_lang_name} to {target_lang_name}. Only return the translation, nothing else:\n\n{text}"
            
            response = self.client.generate_content(prompt)
            translated = response.text.strip()
            
            logger.info(f"Translated text from {source_lang} to {target_lang}")
            return translated
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code
        """
        if not self.client:
            return 'unknown'
        
        try:
            prompt = f"What language is this text in? Only respond with the language name:\n\n{text}"
            response = self.client.generate_content(prompt)
            language = response.text.strip().lower()
            
            # Map to language code
            for code, name in self.SUPPORTED_LANGUAGES.items():
                if name.lower() in language:
                    return code
            
            return 'unknown'
            
        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return 'unknown'
    
    def translate_category_names(self, categories: list, target_lang: str = 'en') -> dict:
        """
        Translate a list of category names
        
        Args:
            categories: List of category names
            target_lang: Target language code
            
        Returns:
            Dictionary mapping original to translated names
        """
        translations = {}
        
        for category in categories:
            translated = self.translate(category, source_lang='pt', target_lang=target_lang)
            translations[category] = translated
        
        return translations
    
    def get_supported_languages(self) -> dict:
        """Get list of supported languages"""
        return self.SUPPORTED_LANGUAGES.copy()

