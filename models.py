import os
import asyncio
import json
from dataclasses import dataclass, field, asdict
from typing import List
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins.openai import LLM as OpenAILLM
import google.generativeai as genai
from enum import Enum
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


@dataclass
class Translation:
    provider: str
    model: str
    language: str
    translation: str


@dataclass
class Sentence:
    text: str
    start: float
    end: float
    translations: List[Translation] = field(default_factory=list)

    def __str__(self) -> str:
        translations_str = "\n  ".join(
            f"{t.language} ({t.provider}/{t.model}): {t.translation}"
            for t in self.translations
        )
        return (
            f"Sentence [{self.start:.2f}s - {self.end:.2f}s]:\n"
            f"  {self.text}\n"
            f"Translations:\n  {translations_str if self.translations else 'None'}"
        )


class LLMProvider(Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class TranslationProvider:
    provider: LLMProvider
    model: str = None


@dataclass
class Transcript:
    sentences: List[Sentence] = field(default_factory=list)

    def __str__(self) -> str:
        return json.dumps(asdict(self), indent=4)


input_languages = {
    "es": "Spanish",
    "en": "English",
    "hi": "Hindi",
    "ja": "Japanese",
    "ru": "Russian",
    "uk": "Ukrainian",
    "sv": "Swedish",
    "zh": "Chinese",
    "pt": "Portuguese",
    "nl": "Dutch",
    "tr": "Turkish",
    "fr": "French",
    "de": "German",
    "id": "Indonesian",
    "ko": "Korean",
    "it": "Italian",
}

output_languages = [
    "Arabic",
    "Simplified Chinese",
    "Traditional Chinese",
    "Dutch",
    "English",
    "French",
    "German",
    "Hebrew",
    "Indonesian",
    "Italian",
    "Japanese",
    "Korean",
    "Polish",
    "Brazilian Portuguese",
    "Portuguese",
    "Latin American Spanish",
    "Spanish",
    "Swedish",
    "Thai",
    "Turkish",
    "Vietnamese",
]


class Engine:
    def __init__(
        self,
        languages: tuple[str, str],
        provider: TranslationProvider,
        system_prompt: str = None,
    ):
        self.input_language, self.output_language = languages
        self.provider = provider

        if system_prompt is None:
            system_prompt = f"You are a translator for the {self.output_language} language. Your only response should be the exact translation of the user input in {self.input_language}, into the {self.output_language} language."

        llm_client_kwargs = {}
        if provider.model is not None:
            llm_client_kwargs["model"] = provider.model

        if provider.provider == LLMProvider.OPENAI:
            self._client = OpenAILLM(**llm_client_kwargs)
            self._chat_ctx = ChatContext(
                messages=[
                    ChatMessage.create(
                        role="system",
                        text=system_prompt,
                    ),
                ]
            )
        elif provider.provider == LLMProvider.GEMINI:
            self._client = genai.GenerativeModel(
                model_name=provider.model,
                system_instruction=system_prompt,
            )
            self._chat_ctx = []
        else:
            raise ValueError(f"Unsupported provider: {provider.provider}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=False,
    )
    async def _translate_gemini(self, sentence: Sentence) -> str:
        # Create temporary context for this attempt
        temp_ctx = self._chat_ctx + [{"role": "user", "parts": sentence.text}]
        stream = await self._client.generate_content_async(
            contents=temp_ctx,
            stream=True,
        )

        response = ""
        try:
            async for chunk in stream:
                response += chunk.text
        except Exception as e:
            print(f"Gemini translation failed (attempt will be retried): {str(e)}")
            raise  # Re-raise the exception to trigger retry

        # Only update the chat context after successful translation
        self._chat_ctx = temp_ctx + [{"role": "model", "parts": response}]
        return response

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=False,
    )
    async def _translate_openai_compat(self, sentence: Sentence) -> str:
        # Create temporary context for this attempt
        temp_ctx = ChatContext(
            messages=self._chat_ctx.messages
            + [ChatMessage.create(role="user", text=sentence.text)]
        )
        stream = self._client.chat(chat_ctx=temp_ctx)

        response = ""
        try:
            async for chunk in stream:
                if chunk.choices:
                    for choice in chunk.choices:
                        if choice.delta and choice.delta.content:
                            response += choice.delta.content
        except Exception as e:
            print(f"OpenAI translation failed (attempt will be retried): {str(e)}")
            raise  # Re-raise the exception to trigger retry

        # Only update the chat context after successful translation
        self._chat_ctx = temp_ctx
        self._chat_ctx.messages.append(
            ChatMessage.create(role="assistant", text=response)
        )
        return response

    async def translate(self, sentence: Sentence) -> str:
        if self.provider.provider == LLMProvider.GEMINI:
            return await self._translate_gemini(sentence)
        else:
            return await self._translate_openai_compat(sentence)


class Translator:
    def __init__(
        self,
        languages: tuple[str, str],
        providers: List[TranslationProvider],
        system_prompt: str = None,
    ):
        self.engines: List[Engine] = []

        for provider in providers:
            self.engines.append(
                Engine(
                    languages=languages,
                    provider=provider,
                    system_prompt=system_prompt,
                )
            )

    async def translate(self, sentence: Sentence) -> Sentence:
        tasks = [engine.translate(sentence) for engine in self.engines]

        # Wait for all translations to complete
        translations = await asyncio.gather(*tasks)

        # translations will be a list of results in the same order as engines
        for i, translation in enumerate(translations):
            engine = self.engines[i]
            sentence.translations.append(
                Translation(
                    provider=engine.provider.provider.value,
                    model=engine.provider.model,
                    language=engine.output_language,
                    translation=translation,
                )
            )

        return sentence
