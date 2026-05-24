import logging
import os

import torch

import mteb
from mteb.cache import ResultCache
from mteb.models import SearchEncoderWrapper

logger = logging.getLogger(__name__)

os.environ["HF_HOME"] = ".cache"
logging.basicConfig(level=logging.INFO)

cache = ResultCache("result_cache")
# cache.download_from_remote()


class DataParallelHF(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


for model_name in [
    # "google/embeddinggemma-300m",
    # "intfloat/multilingual-e5-large-instruct",
    # "microsoft/harrier-oss-v1-270m",
    # "intfloat/multilingual-e5-small",
    # "intfloat/multilingual-e5-base",
    # "intfloat/multilingual-e5-large",
    # "microsoft/harrier-oss-v1-0.6b",
    # "Qwen/Qwen3-Embedding-8B",
    # "Qwen/Qwen3-Embedding-4B",
    # "perplexity-ai/pplx-embed-v1-0.6b",
    # "perplexity-ai/pplx-embed-v1-4b",
    # "perplexity-ai/pplx-embed-v1-8b",
    "BidirLM/BidirLM-270M-Embedding",
    "BidirLM/BidirLM-0.6B-Embedding",
    "BidirLM/BidirLM-1.7B-Embedding",  # !!!!!!!!!!!!!!!!!!!!!!!!
    # "NovaSearch/stella_en_400M_v5",
    # "NovaSearch/stella_en_1.5B_v5",
    # "NovaSearch/jasper_en_vision_language_v1",
    # "tencent/KaLM-Embedding-Gemma3-12B-2511",
    # "BAAI/bge-m3",
    # "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v2",
    # "KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5",
    # "codefuse-ai/F2LLM-v2-80M",
    # "codefuse-ai/F2LLM-v2-160M",
    # "codefuse-ai/F2LLM-v2-330M",
    # "codefuse-ai/F2LLM-v2-0.6B",
    # "codefuse-ai/F2LLM-v2-4B",
    # "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    # "sentence-transformers/LaBSE",
    # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # "sentence-transformers/static-similarity-mrl-multilingual-v1",
    # "sentence-transformers/all-MiniLM-L12-v2",
    # "sentence-transformers/all-MiniLM-L6-v2",
    # "sentence-transformers/all-mpnet-base-v2",
    # "nvidia/llama-embed-nemotron-8b",
    # "princeton-nlp/sup-simcse-bert-base-uncased",
    # "princeton-nlp/unsup-simcse-roberta-base",
    # "princeton-nlp/unsup-simcse-bert-large-uncased",
    # "princeton-nlp/unsup-simcse-roberta-large",
    # "princeton-nlp/sup-simcse-bert-large-uncased",
    # "princeton-nlp/sup-simcse-roberta-base",
    # "princeton-nlp/sup-simcse-roberta-large",
    "TODBERT/TOD-BERT-MLM-V1",
    "AndrewZeng/futuretod-base-v1.0",
]:
    try:
        # model = mteb.get_model(model_name, processor_kwargs={"model_max_length": 8_192})
        # model.mteb_model_meta.experiment_kwargs = None
        model = mteb.get_model(model_name)
    except Exception as e:
        print(e)
        continue
    # model.model.max_seq_length = 8192
    model.model[0].auto_model = DataParallelHF(model.model[0].auto_model)
    model = SearchEncoderWrapper(model)

    for task in mteb.get_benchmark("DialogMTEB(v1)"):
        for batch_size in [
            # 2048,
            # 1024,
            512,
            128,
            64,
            32,
            16,
            8,
            4,
            2,
            1,
        ]:
            try:
                mteb.evaluate(
                    model,
                    task,
                    cache=cache,
                    encode_kwargs={"batch_size": batch_size * 8},
                    # overwrite_strategy="always",
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"CUDA OOM on batch size {batch_size}")
                logger.warning(f"CUDA OOM on batch size {batch_size}")
                continue
            except Exception as e:
                print(f"Got excpetion {e}")
                logger.error(f"Got excpetion {e}")
                raise
                break
            else:
                break
