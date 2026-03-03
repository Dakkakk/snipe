import asyncio
import logging
import os

import path
import tqdm.auto
from gradio import Error
from pdf2zh_next import high_level, SettingsModel, BasicSettings, TranslationSettings, PDFSettings, OllamaSettings
from babeldoc.assets import assets
from pdf2zh_next.high_level import TranslationError
from tqdm.auto import trange

custom_system_prompt = """You are a professional English (en) to Chinese (zh-Hans) Music translator. Your goal is to accurately convey the meaning and nuances of the original English text while adhering to Chinese grammar, vocabulary, and cultural sensitivities.
Produce only the Chinese translation, without any additional explanations or commentary. Please translate the following English text from a book about teaching piano into Chinese:

"""

LOG=logging.getLogger(__name__)

async def pdfGen(files: list):
    settings = SettingsModel(
        basic=BasicSettings(
            debug=False,
        ),
        translation=TranslationSettings(
            custom_system_prompt=custom_system_prompt,
            lang_in="en",
            lang_out="zh",
            output="./files_out",
            no_auto_extract_glossary=True,
            save_auto_extracted_glossary=False,
        ),
        pdf=PDFSettings(
            watermark_output_mode="no_watermark",
            translate_table_text=False,
            split_short_lines=True,
            only_include_translated_page=True,
            use_alternating_pages_dual=True,
        ),
        translate_engine_settings=OllamaSettings(
            support_llm="no",
            ollama_model="translategemma:27b",
            # ollama_host="8.149.241.189:20662"
        ),
        term_extraction_engine_settings=None
    )
    fileProgress = tqdm.auto.tqdm(files, desc="PDF Generation", )

    progresses = {}
    for file in fileProgress:
        fileProgress.postfix = f"{fileProgress.n}/{fileProgress.total}({file}"
        LOG.info(f"start file: {file}")
        for progress in progresses.values():
            progress.close()
        progresses.clear()
        try:
            async for event in high_level.do_translate_async_stream(settings, os.path.join("files_in", file)):
                if event["type"] in (
                        "progress_start",
                        "progress_update",
                        "progress_end",
                ):
                    # Update progress bar
                    desc = event["stage"]
                    if not progresses.__contains__(desc):
                        progresses[desc] = tqdm.auto.tqdm(total=100.0, initial=0.0)
                    progress: tqdm.auto.tqdm = progresses[desc]
                    part_index = event["part_index"]
                    total_parts = event["total_parts"]
                    progress_value = event["overall_progress"]
                    stage_current = event["stage_current"]
                    stage_total = event["stage_total"]
                    progress.desc = desc
                    progress.postfix = f"({part_index}/{total_parts}, {stage_current}/{stage_total})"
                    progress.n = stage_current
                    progress.total=stage_total

                    progress.update(0)
                elif event["type"] == "finish":
                    # Extract result paths
                    result = event["translate_result"]
                    mono_path = result.mono_pdf_path
                    dual_path = result.dual_pdf_path
                    glossary_path = result.auto_extracted_glossary_path
                    token_usage = event.get("token_usage", {})
                    LOG.info(f"finished {mono_path}")
                    LOG.info(f"token usage: {token_usage}")
                    break
                elif event["type"] == "error":
                    # Handle error event
                    error_msg = event.get("error", "Unknown error")
                    error_details = event.get("details", "")
                    raise Error(f"Translation error: {error_msg}")
        except asyncio.CancelledError:
            LOG.error(f"a task was cancelled{file}")
        except TranslationError as e:
            LOG.error(f"Translation error: {file}:{e}/{e.with_traceback(None)}")
        except Exception as e:
            LOG.error(f"error occurred:{file}:{e}/{e.with_traceback(None)}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    from tqdm.contrib.logging import logging_redirect_tqdm
    with logging_redirect_tqdm():
        files = sorted(os.listdir("files_in"))
        LOG.info(f"files: {files}")
        asyncio.run(pdfGen(files))
