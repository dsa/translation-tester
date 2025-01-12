import os
import json
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    PrerecordedResponse,
    FileSource,
    Sentence as DeepgramSentence,
)

from models import (
    Sentence,
    Transcript,
    Translation,
    Translator,
    TranslationProvider,
    LLMProvider,
    input_languages,
    output_languages,
)
from typing import List
from sqlalchemy.orm import Session
from database import init_db, DBAudioFile, DBSentence, DBTranslation, SessionLocal

providers = [
    TranslationProvider(provider=LLMProvider.OPENAI, model="gpt-4o"),
    TranslationProvider(provider=LLMProvider.GEMINI, model="gemini-1.5-pro"),
]


async def translate(
    dg_sentence: DeepgramSentence,
    translators: List[Translator],
    db_audio: DBAudioFile,
    session: Session,
) -> Sentence:
    # Check if sentence already exists in DB
    db_sentence = (
        session.query(DBSentence)
        .filter_by(
            audio_file_id=db_audio.id,
            text=dg_sentence.text,
            start=dg_sentence.start,
            end=dg_sentence.end,
        )
        .first()
    )

    if db_sentence is None:
        # Create new sentence in DB
        db_sentence = DBSentence(
            audio_file_id=db_audio.id,
            text=dg_sentence.text,
            start=dg_sentence.start,
            end=dg_sentence.end,
        )
        session.add(db_sentence)
        session.commit()

    sentence = Sentence(**dg_sentence.to_dict())

    # Load existing translations from database and clean them up
    for db_translation in db_sentence.translations:
        # Clean up existing translation in the database
        if "\n" in db_translation.translation:
            db_translation.translation = " ".join(
                db_translation.translation.split()
            )  # Removes all extra whitespace and newlines
            session.add(db_translation)
            session.commit()

        sentence.translations.append(
            Translation(
                provider=db_translation.provider,
                model=db_translation.model,
                language=db_translation.language,
                translation=db_translation.translation,
            )
        )

    # Check which translations are missing
    existing_translations = {(t.language, t.provider) for t in db_sentence.translations}

    for translator in translators:
        for engine in translator.engines:
            key = (engine.output_language, engine.provider.provider.value)
            if key not in existing_translations:
                translation = await engine.translate(sentence)
                # Clean up the translation text
                translation = translation.strip()

                db_translation = DBTranslation(
                    sentence_id=db_sentence.id,
                    provider=engine.provider.provider.value,
                    model=engine.provider.model,
                    language=engine.output_language,
                    translation=translation,
                )
                session.add(db_translation)
                session.commit()

                sentence.translations.append(
                    Translation(
                        provider=engine.provider.provider.value,
                        model=engine.provider.model,
                        language=engine.output_language,
                        translation=translation,
                    )
                )

    print(sentence)
    return sentence


def transcribe(audio_path: str, language_override: str = None) -> PrerecordedResponse:
    try:
        deepgram = DeepgramClient()

        with open(audio_path, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            filler_words=True,
            detect_language=language_override is None,  # Only detect if no override
        )

        if language_override:
            options.language = language_override

        return deepgram.listen.rest.v("1").transcribe_file(
            payload, options, timeout=500
        )

    except Exception:
        raise


def get_file_paths(audio_path: str) -> tuple[str, str, str]:
    audio_path = Path(audio_path)
    base_name = audio_path.stem

    # Generate timestamp string only for translation file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Transcript file without timestamp
    transcript_path = Path("files/transcripts") / f"{base_name}.json"
    # Translation file with timestamp
    translation_path = Path("files/translations") / f"{base_name}_{timestamp}.json"

    # Create directories (this won't overwrite files)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    translation_path.parent.mkdir(parents=True, exist_ok=True)

    return str(audio_path), str(transcript_path), str(translation_path)


async def main():
    parser = argparse.ArgumentParser(description="Transcribe and translate audio file")
    parser.add_argument(
        "--audio-file",
        required=True,
        help="Path to the audio file to process",
        type=str,
    )
    parser.add_argument(
        "--concurrent-sentences",
        type=int,
        default=5,
        help="Number of sentences to process concurrently",
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Override source language (e.g., 'en', 'es', 'fr'). If not provided, language will be auto-detected.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        parser.error(f"Audio file not found: {args.audio_file}")

    init_db()
    session = SessionLocal()

    try:
        audio_path, transcript_path, translation_path = get_file_paths(args.audio_file)

        # Get filename instead of full path
        filename = Path(audio_path).name

        # Get or create audio file record once at the start
        db_audio = session.query(DBAudioFile).filter_by(filename=filename).first()
        if db_audio is None:
            db_audio = DBAudioFile(filename=filename)
            session.add(db_audio)
            session.commit()

        if os.path.exists(transcript_path):
            with open(transcript_path, "r") as file:
                response = PrerecordedResponse.from_json(json.load(file))
        else:
            print("Starting transcription process...")
            response = transcribe(audio_path, args.language)
            with open(transcript_path, "w") as file:
                json.dump(response.to_json(), file, indent=4)

        paragraphs = response.results.channels[0].alternatives[0].paragraphs

        # Get the source language - either from detection or override
        source_language = (
            args.language
            if args.language
            else response.results.channels[0].detected_language
        )

        translators = []
        for output_language in output_languages:
            translators.append(
                Translator(
                    languages=(
                        input_languages[source_language],
                        output_language,
                    ),
                    providers=providers,
                )
            )

        transcript = Transcript()

        print("Starting translation process...")
        # Create tasks for all sentences
        tasks = []
        for paragraph in paragraphs.paragraphs:
            for sentence in paragraph.sentences:
                # Create task but don't await it yet
                task = translate(
                    sentence,
                    translators,
                    db_audio,
                    session,
                )
                tasks.append(task)

        # Process sentences in parallel, with a concurrency limit
        chunk_size = 5  # Adjust this number based on your needs
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i : i + chunk_size]
            chunk_results = await asyncio.gather(*chunk)
            for translation in chunk_results:
                transcript.sentences.append(translation)

        with open(translation_path, "w") as file:
            file.write(str(transcript))

    finally:
        session.close()


if __name__ == "__main__":
    asyncio.run(main())
