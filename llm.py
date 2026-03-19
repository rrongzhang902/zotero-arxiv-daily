from llama_cpp import Llama
from openai import OpenAI
from loguru import logger
from time import sleep
import requests

GLOBAL_LLM = None
DEFAULT_VOLCENGINE_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
DEFAULT_VOLCENGINE_TRANSLATION_MODEL = "doubao-seed-2-0-lite-260215"

class LLM:
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
        lang: str = "English",
        use_volcengine_translation: bool = True,
        volcengine_api_key: str = None,
        volcengine_base_url: str = DEFAULT_VOLCENGINE_BASE_URL,
        volcengine_translation_model: str = DEFAULT_VOLCENGINE_TRANSLATION_MODEL,
    ):
        if api_key:
            self.llm = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.llm = Llama.from_pretrained(
                repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
                filename="qwen2.5-3b-instruct-q4_k_m.gguf",
                n_ctx=5_000,
                n_threads=4,
                verbose=False,
            )
        self.model = model
        self.lang = lang
        self.use_volcengine_translation = use_volcengine_translation
        self.volcengine_api_key = volcengine_api_key
        self.volcengine_base_url = volcengine_base_url
        self.volcengine_translation_model = volcengine_translation_model

    def generate(self, messages: list[dict]) -> str:
        if isinstance(self.llm, OpenAI):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.llm.chat.completions.create(messages=messages, temperature=0, model=self.model)
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    sleep(3)
            return response.choices[0].message.content
        else:
            response = self.llm.create_chat_completion(messages=messages,temperature=0)
            return response["choices"][0]["message"]["content"]

    def _translate_with_volcengine(self, text: str, target: str) -> str:
        payload = {
            "model": self.volcengine_translation_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant who accurately translates scientific writing. "
                        "Return only the translated text without explanations, markdown, or extra labels."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Translate the following scientific summary into {target}. "
                        "Keep it concise, accurate, and preserve technical terminology. "
                        "Return only the translation.\n\n"
                        f"{text}"
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 300,
        }
        headers = {
            "Authorization": f"Bearer {self.volcengine_api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            self.volcengine_base_url,
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def translate(self, text: str, target_lang: str = None) -> str:
        target = target_lang or self.lang
        if self.use_volcengine_translation:
            if self.volcengine_api_key:
                try:
                    return self._translate_with_volcengine(text, target)
                except Exception as e:
                    logger.warning(
                        f"Volcengine translation failed, falling back to default translator: {e}"
                    )
            else:
                logger.warning(
                    "USE_VOLCENGINE_TRANSLATION is enabled but VOLCENGINE_API_KEY is not set. Falling back to default translator."
                )
        prompt = (
            f"Translate the following scientific summary into {target}. "
            "Keep it concise, accurate, and preserve technical terminology.\n\n"
            f"{text}"
        )
        return self.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant who accurately translates scientific writing.",
                },
                {"role": "user", "content": prompt},
            ]
        )

def set_global_llm(
    api_key: str = None,
    base_url: str = None,
    model: str = None,
    lang: str = "English",
    use_volcengine_translation: bool = True,
    volcengine_api_key: str = None,
    volcengine_base_url: str = DEFAULT_VOLCENGINE_BASE_URL,
    volcengine_translation_model: str = DEFAULT_VOLCENGINE_TRANSLATION_MODEL,
):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(
        api_key=api_key,
        base_url=base_url,
        model=model,
        lang=lang,
        use_volcengine_translation=use_volcengine_translation,
        volcengine_api_key=volcengine_api_key,
        volcengine_base_url=volcengine_base_url,
        volcengine_translation_model=volcengine_translation_model,
    )

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM
