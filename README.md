# Translation Tester

This Python program takes a pre-recorded audio file, sends it to Deepgram to transcribe into text, and then sentence-by-sentence, feeds the transcript into any number of LLMs for translation into any number of languages. 

The code here has basic error handling and retry logic. For large audio files or lengthy transcripts, you may encounter bugs, issues or rate limits with the LLMs you use. This program uses SQLite to store translations as they're generated in case something goes haywire and crashes. You can always rerun the program from scratch to fill in missing translations in the event of an error. Existing database records will _not_ be overwritten.

To translate into a new language or to add another LLM provider, you can modify the code in `main.py`:
```
providers = [
    TranslationProvider(provider=LLMProvider.OPENAI, model="gpt-4o"),
    TranslationProvider(provider=LLMProvider.GEMINI, model="gemini-1.5-pro"),
]
```

> [!NOTE]  
> Translations are done in batches of 5 concurrently. You can change this via CLI argument passed to the program when you run the script with --concurrent-sentences.
> 
> The program also has a convention where audio input files are stored under `/files/audio`, transcriptions are stored (to avoid re-transcribing) under `/files/transcripts` and translations are stored under `/files/translations`

## How to run this code:
1. `python -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. `cp .env.example .env`
5. add values for keys in `.env`
6. `python main.py --audio-file [path_to_mp3_file]`
