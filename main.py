# from processors.marker.pipeline import pipeline
# from processors.tika.file_processor import FileProcessor
# from processors.elasticsearch_indexer import ElasticsearchIndexer
# from processors.chunkers import Chunker
# from processors.embedder import EmbeddingHandler
from evaluation.pipeline import run_evaluation_pipeline
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import asyncio
import os
import shutil
from tqdm import tqdm
import yaml
from config_logger import logger
import json
import itertools

load_dotenv()


def check_index_exists(index_name):
    es = Elasticsearch(os.getenv("ES_URL", "http://localhost:9200"))
    return es.indices.exists(index=index_name)


class IndexingPipeline:
    def __init__(self, folder_path, processor, processing_mode, use_vlm, vlm_name, chunker, chunk_size, embedder):
        self.folder_path = folder_path
        self.processor = processor
        self.processing_mode = processing_mode
        self.use_vlm = use_vlm
        self.vlm_name = vlm_name
        self.chunker = chunker
        self.chunk_size = chunk_size
        self.embedder = embedder

    def index_files(self):
        index_name = self.index_name_generator()

        if check_index_exists(index_name):
            return index_name

        indexer = ElasticsearchIndexer(index_name)
        embedder = EmbeddingHandler(self.embedder)
        if self.processor == "marker":
            for file in tqdm(os.listdir(self.folder_path)):
                if file.endswith(".pdf"):
                    file_path = os.path.join(self.folder_path, file)
                    pipeline(file_path)
                    output_md = os.path.join("./processors/marker/output", file.strip(".pdf"), file.strip(".pdf") + "_out.md")
                    chunker = Chunker(self.chunker, self.chunk_size, output_md)
                    chunks = chunker.create_chunks()
                    indexer.index_data(chunks, embedder.get_embeddings())
                    shutil.rmtree("./processors/marker/output")
            return index_name
        elif self.processor == "tika":
            for file in tqdm(os.listdir(self.folder_path)):
                if file.endswith(".pdf"):
                    file_path = os.path.join(self.folder_path, file)
                    tika_processor = FileProcessor(file_path, self.processing_mode, index_name, use_vlm=True)
                    output_file_path = tika_processor.process_file()
                    if output_file_path:
                        chunker = Chunker(self.chunker, self.chunk_size, output_file_path)
                        chunks = chunker.create_chunks()
                        indexer.index_data(chunks, embedder.get_embeddings())
                        shutil.rmtree("./processors/tika/output")
            return index_name

    def index_name_generator(self):
        for file in os.listdir(self.folder_path):
            if file.endswith(".pdf"):
                index_name = "testIndex_" + str(self.folder_path).split("/")[-1] + "_" + self.processor + "_" + self.vlm_name.replace("/", "") \
                            + "_" + self.chunker + "_" + str(self.chunk_size) + "size_" + self.embedder.replace("/", "")
                return index_name.lower()


def run_retrieval_test(config, test_indices):
    search_methods = config.get("search_methods", [])
    retriever_weights_options = config.get("retriever_weights", [])
    reranked_options = config.get("reranked_options", [])
    ground_truth_path = config.get("ground_truth_path", "./data/ground_truth/qa_dataset.csv")
    do_test_retrieval = config.get("test_retrieval", True)
    do_test_generation = config.get("test_generation", False)

    logger.warning("=" * 50)
    logger.warning("Starting retrieval tests")
    logger.warning(f"Search Methods: {search_methods}")
    logger.warning(f"Retriever Weights: {retriever_weights_options}")
    logger.warning(f"Reranked Options: {reranked_options}")
    logger.warning(f"Test Indices: {test_indices}")
    logger.warning("=" * 50)

    best_configs = {}

    for index_name in test_indices:
        logger.warning(f"Starting tests for index: {index_name}")

        best_configs[index_name] = {
            "index_name": index_name,
            "search_method": None,
            "retriever_weights": None,
            "reranked": None,
            "mrr": 0,
            "generation_score": 0
        }

        for search_method in search_methods:
            for reranked in reranked_options:
                weights_list = retriever_weights_options if search_method in ["hybrid", "hybrid_cc"] else [[1.0, 0.0]]
                for retriever_weights in weights_list:
                    try:
                        logger.warning(
                            f"Testing: index={index_name}, method={search_method}, weights={retriever_weights}, reranked={reranked}"
                        )
                        final_mrr, avg_generation_score = asyncio.run(
                            run_evaluation_pipeline(
                                ground_truth_path, index_name, retriever_weights,
                                search_method, reranked, do_test_retrieval, do_test_generation,
                            )
                        )
                        logger.warning(f"MRR: {final_mrr}, Generation Score: {avg_generation_score}")

                        if final_mrr > best_configs[index_name]["mrr"]:
                            best_configs[index_name].update({
                                "search_method": search_method,
                                "retriever_weights": retriever_weights,
                                "reranked": reranked,
                                "mrr": final_mrr
                            })
                    except Exception as e:
                        logger.error(f"Error during retrieval testing for index={index_name}: {e}")

    logger.warning("Retrieval testing completed.")
    return best_configs


def run_generation_test(config, test_indices):
    search_methods = config.get("search_methods", [])
    retriever_weights_options = config.get("retriever_weights", [])
    reranked_options = config.get("reranked_options", [])
    ground_truth_path = config.get("ground_truth_path", "./data/ground_truth/qa_dataset.csv")
    do_test_retrieval = config.get("test_retrieval", False)
    do_test_generation = config.get("test_generation", True)

    generation_scores = {}

    for index_name in test_indices:
        for search_method in search_methods:
            for reranked in reranked_options:
                weights_list = retriever_weights_options if search_method in ["hybrid", "hybrid_cc"] else [[1.0, 0.0]]
                for retriever_weights in weights_list:
                    try:
                        final_mrr, avg_generation_score, model_name = asyncio.run(
                            run_evaluation_pipeline(
                                ground_truth_path, index_name, retriever_weights,
                                search_method, reranked, do_test_retrieval, do_test_generation,
                            )
                        )
                        generation_scores[model_name] = avg_generation_score
                    except Exception as e:
                        logger.error(f"Error during generation testing for index={index_name}: {e}")

    with open("generation_scores.json", "w") as f:
        json.dump(generation_scores, f, indent=4)

    logger.warning("Generation scores saved to generation_scores.json.")
    return generation_scores


def save_best_configs(best_configs):
    best_config_file = "best_configs.json"
    try:
        existing_configs = {}
        if os.path.exists(best_config_file):
            with open(best_config_file, "r") as f:
                existing_configs = json.load(f)

        existing_configs.update(best_configs)
        with open(best_config_file, "w") as f:
            json.dump(existing_configs, f, indent=4)

        logger.warning(f"Best configs saved to {best_config_file}")
    except Exception as e:
        logger.error(f"Error saving best configs: {e}")


def create_index(folder_path, file_processor, processing_mode, use_vlm, vlm_name, chunk_method, chunk_size, embedder_name):
    pipeline = IndexingPipeline(
        folder_path, file_processor, processing_mode,
        use_vlm, vlm_name, chunk_method, chunk_size, embedder_name
    )
    return pipeline.index_files()


def make_indexes(config):
    new_indices = []
    chunk_methods = config['chunk_method']
    chunk_sizes = config['chunk_size']
    file_processors = config['file_processor']
    processing_modes = config['processing_mode']
    data_path = config['data_path']

    for chunk_method, chunk_size, file_processor, processing_mode in itertools.product(chunk_methods, chunk_sizes, file_processors, processing_modes):
        try:
            index_name = create_index(
                folder_path=data_path,
                file_processor=file_processor,
                processing_mode=processing_mode,
                use_vlm=config['use_vlm'],
                vlm_name=config['vlm_name'],
                chunk_method=chunk_method,
                chunk_size=chunk_size,
                embedder_name=config['embedder_name']
            )
            logger.warning(f"Indexed: {index_name} (chunker={chunk_method}, size={chunk_size}, processor={file_processor})")
            new_indices.append(index_name)
        except Exception as e:
            logger.error(f"Error indexing (chunker={chunk_method}, size={chunk_size}, processor={file_processor}): {e}")
            raise

    return new_indices


def delete_indices(indices):
    es = Elasticsearch(os.getenv("ES_URL", "http://localhost:9200"))
    for index in indices:
        es.indices.delete(index=index)


if __name__ == "__main__":
    try:
        with open("config.yaml", "r") as config_file:
            config = yaml.safe_load(config_file)
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        raise
    logger.warning("Loaded configuration successfully.")

    only_test_indices = config.get("only_test_index", False)
    test_indices = config.get("test_indices", [])
    do_test_generation = config.get("test_generation", False)
    do_test_retrieval = config.get("test_retrieval", True)

    if only_test_indices:
        if not test_indices:
            logger.error("No test indices specified in the configuration.")
            raise ValueError("Test indices are required when `only_test_index` is true.")
        logger.warning("Skipping index creation. Testing specified indices.")

        if do_test_generation:
            best_configs = run_generation_test(config, test_indices)
        elif do_test_retrieval:
            best_configs = run_retrieval_test(config, test_indices)
        save_best_configs(best_configs)

    else:
        new_indices = make_indexes(config)

        if do_test_generation:
            best_configs = run_generation_test(config, test_indices)
        elif do_test_retrieval:
            best_configs = run_retrieval_test(config, test_indices)
            save_best_configs(best_configs)

        if do_test_generation:
            best_config_new_indices = run_generation_test(config, new_indices)
        elif do_test_retrieval:
            best_config_new_indices = run_retrieval_test(config, new_indices)
            save_best_configs(best_config_new_indices)
