import hashlib
import json
import os
from pydantic import BaseModel, Field, ValidationError, RootModel
from typing import Dict, List
from basic_agent import BasicAgent
from cosmos_client import SimpleCosmosClient
# from dotenv import load_dotenv

# load_dotenv(override=True)


COSMOS_CONNECTION_STRING = os.environ.get("COSMOS_CONNECTION_STRING")
COSMOS_DATABASE_NAME = os.environ.get("COSMOS_DATABASE_NAME")
PARTITION_KEY_PATH = "/id"


StrList = List[str]


class Summary(BaseModel):
    news: str
    keywords: StrList = Field(alias="keywords")
    companies: StrList = Field(alias="companies")
    model_name: StrList = Field(alias="model name")
    model_architecture: StrList = Field(alias="model architecture")
    detailed_model_version: StrList = Field(alias="detailed model version")
    ai_tools: StrList = Field(alias="ai tools")
    infrastructure: StrList = Field(alias="infrastucture")
    ml_techniques: StrList = Field(alias="ml techniques")


class Payload(RootModel):
    root: Dict[str, Summary]  # e.g. {"news summary 1": Summary, …}


def query_llm(
    prompt: str,
    agent,
    model: str = "gemini-2.0-flash-exp",
    max_tries: int = 3,
) -> Dict[str, Summary]:
    p = prompt
    for _ in range(max_tries):
        print(f"attempt {_ + 1} of {max_tries}")
        # txt = agent.get_text_response_from_llm(model, messages=p, code_tag=None)[
        txt = agent._get_text_response_from_llm(
            "priv_openai:gpt-4.1", messages=p, code_tag=None
        )["text_response"]
        if "```" in txt:
            parts = txt.split("```")
            if len(parts) >= 3:
                txt = parts[1]
                if txt.lstrip().startswith("json"):
                    txt = txt[4:].lstrip()
        try:
            data = json.loads(txt)
            validated = Payload.model_validate(data)
            return validated.root
        except (json.JSONDecodeError, ValidationError) as err:
            p = f"{txt}\n\nYour JSON was invalid ({err}). Fix it and return only valid JSON."
    raise RuntimeError("Failed to obtain valid payload after 3 attempts.")


def make_piece_id(parent_id: str, headline: str) -> str:
    digest = hashlib.sha1(headline.encode("utf-8")).hexdigest()[:12]
    return f"{parent_id}_{digest}"


def process_chunk(pieces_container_client, chunk_to_do):
    """
    Process a chunk of text and extract structured information.
    Args:
        chunk_to_do (dict): The chunk of text to process.
    Returns:
        bool: True if processing was successful, False otherwise.
    """

    # these keys are copied from chunk to piece
    # and are not modified
    keys_to_copy = [
        "id",
        "source",
        "chunk_date",
        "processing_target_date",
    ]

    message_template = """This is a content of last newslettters about AI: {text}. #####
        Your task is to extract news and opinions about AI related topics from the text. Accompany it with keywords organized into topics:
        {{
            "... news headline ..." : {{
                "news": "Gemini 2.0 has been released with new features.",
                "keywords": ["", ""], # one of: "LLMs", "AI Agents", "AI Infrastructure", "AI Engineering & Tooling", "AI Applications", "Benchmarks", "Metrics", "AI Companies", "AI Researchers & Engineers", "AI Communities", "AI Events", "AI Governance", "AI Ethics", "AI Bias", "AI Impact on Work", "Copyright", "Datasets", "Model Weights", "Technical Reports", "Open Source Projects",
                "companies": ["", ""], # e.g. "Google", "Anthropic", "OpenAI", "Meta", "Tencent", "DeepSeek", "Perplexity AI", "Cartesia", "PrimeIntellect", "Alibaba", "HuggingFace", "Unsloth AI", "Nous Research AI",
                "model name": ["", ""], #  e.g. "Gemini 2.0", "Claude 3", "Claude Code", "Gemini 2.5 Pro", "Gemini 2.5 Turbo",
                "model architecture": ["", ""], # e.g. "Transformer", "MoE", "Llama", "Claude", "Gemini", "DeepSeek", "Grok", "Sonar",
                "detailed model version": ["", ""], # e.g. "Gemini 2.0", "Claude 3", "Claude Code", "Gemini 2.5 Pro", "Gemini 2.5 Turbo",
                "ai tools: ["", ""], # e.g. "KerasRS", "LangChain", "LlamaIndex", "Aider", "Cursor", "Windsurf", "CUTLASS", "CuTe DSL", "Torchtune", "Mojo",
                "infrastucture": ["", ""], # e.g. "NVIDIA", "Intel Arc", "TPUs", "VRAM", "KV Cache", "CUDA", "IPEX", "SYCL",
                "ml techniques": ["", ""] # e.g. "XGBoost", "RL", "Supervised Learning", "Post-Training", "Quantization Techniques", "Inference Optimization
            }}
        }}   
    """

    try:
        print(f"Processing chunk: {chunk_to_do['id']}")
        message = message_template.format(text=chunk_to_do["text"])
        cleaned_payload = query_llm(message, BasicAgent(), model="gemini-2.0-flash-exp")
        pieces_to_paste = {k: v for k, v in chunk_to_do.items() if k in keys_to_copy}
        for headline, summary in cleaned_payload.items():
            piece_to_paste = pieces_to_paste.copy()
            piece_to_paste["parent_id"] = chunk_to_do["id"]
            piece_to_paste["id"] = make_piece_id(chunk_to_do["id"], headline)
            piece_to_paste["headline"] = headline
            for k, v in summary.model_dump().items():
                if k != "id":
                    piece_to_paste[k] = v
            pieces_container_client.upsert_item(piece_to_paste)
        return True
    except Exception as e:
        print(f"Error processing chunk {chunk_to_do['id']}: {e}")
        return False


def main():
    # Initialize Cosmos Client

    try:
        cosmos_client = SimpleCosmosClient(
            connection_string=COSMOS_CONNECTION_STRING,
            database_name=COSMOS_DATABASE_NAME,
            partition_key_path=PARTITION_KEY_PATH,
        )

        cosmos_client.connect()
        pieces = cosmos_client.database_client.get_container_client("knowledge-pieces")
        chunks = cosmos_client.database_client.get_container_client("knowledge-chunks")
    except Exception as e:
        raise  # Critical failure, cannot proceed without DB access

    done_ids = list(
        pieces.query_items(
            query="SELECT VALUE p.parent_id FROM p",
            enable_cross_partition_query=True,
        )
    )

    query_to_get_a_note = """
        SELECT TOP 1000 *
        FROM c
        WHERE NOT ARRAY_CONTAINS(@done, c.id)
        ORDER BY c.chunk_date  
    """

    chunks_to_do = list(
        chunks.query_items(
            query=query_to_get_a_note,
            parameters=[{"name": "@done", "value": done_ids}],
            enable_cross_partition_query=True,
        )
    )

    print(f"len chunks_to_do: {len(chunks_to_do)}")

    if len(chunks_to_do) > 0:
        for chunk_to_do in chunks_to_do:
            process_chunk(pieces, chunk_to_do)
    else:
        print("No chunks to process.")

    return None


if __name__ == "__main__":
    main()
