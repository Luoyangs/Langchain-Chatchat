import os
import shutil
from typing import List, Dict, Tuple

import numpy as np
from langchain.docstore.document import Document

from configs import SCORE_THRESHOLD
from server.knowledge_base.kb_cache.faiss_cache import kb_faiss_pool, ThreadSafeFaiss
from server.knowledge_base.kb_service.base import KBService, SupportedVSType, EmbeddingsFunAdapter
from server.knowledge_base.utils import KnowledgeFile, get_kb_path, get_vs_path
from server.utils import torch_gc


class FaissKBService(KBService):
    vs_path: str
    kb_path: str
    vector_name: str = None
 
    def vs_type(self) -> str:
        return SupportedVSType.FAISS

    def get_vs_path(self):
        return get_vs_path(self.kb_name, self.vector_name)

    def get_kb_path(self):
        return get_kb_path(self.kb_name)

    def load_vector_store(self) -> ThreadSafeFaiss:
        return kb_faiss_pool.load_vector_store(kb_name=self.kb_name,
                                               vector_name=self.vector_name,
                                               embed_model=self.embed_model)

    def save_vector_store(self):
        self.load_vector_store().save(self.vs_path)

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        with self.load_vector_store().acquire() as vs:
            return [vs.docstore._dict.get(id) for id in ids]

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        with self.load_vector_store().acquire() as vs:
            vs.delete(ids)

    def do_init(self):
        self.vector_name = self.vector_name or self.embed_model
        self.kb_path = self.get_kb_path()
        self.vs_path = self.get_vs_path()

    def do_create_kb(self):
        if not os.path.exists(self.vs_path):
            os.makedirs(self.vs_path)
        self.load_vector_store()

    def do_drop_kb(self):
        self.clear_vs()
        try:
            shutil.rmtree(self.kb_path)
        except Exception:
            ...

    def do_search(self,
                  query: str,
                  top_k: int,
                  score_threshold: float = SCORE_THRESHOLD,
                  ) -> List[Tuple[Document, float]]:
        embed_func = EmbeddingsFunAdapter(self.embed_model)
        embeddings = embed_func.embed_query(query)
        with self.load_vector_store().acquire() as vs:
            docs = vs.similarity_search_with_score_by_vector(embeddings, k=top_k, score_threshold=score_threshold)
            doc_lens = len(docs)
            if doc_lens == 0:
                return docs

            results = []
            embedding = vs.embedding_function.embed_query(query)
            vector = np.array([embedding], dtype=np.float32)
            scores, indices = vs.index.search(vector, doc_lens)
            kb_files = set([d.metadata.get("source") for d, s in docs])
            name_map_id = dict()
            for i, doc in vs.docstore._dict.items():
                name = doc.metadata.get("source")
                if name in kb_files:
                    if name not in name_map_id:
                        name_map_id[name] = [i]
                    else:
                        name_map_id[name].append(i)
            # 保存最终的文档id, 这是个不重复的数组
            doc_ids = []
            doc_scores = []
            for j, i in enumerate(indices[0]):
                _id = vs.index_to_docstore_id[i]
                res_score = docs[j][1]
                if _id not in doc_ids:
                    doc_ids.append(_id)
                    doc_scores.append(res_score)

                doc = vs.docstore.search(_id)
                doc_source_name = doc.metadata.get("source")
                if doc_source_name in name_map_id:
                    id_list = name_map_id[doc_source_name]
                    id_index = id_list.index(_id)
                    id_len = 5 - j * 2  # 按照5:3:1的数量来扩充context
                    if id_len <= 0:
                        break
                    id_ava = id_list[id_index + 1:id_index + 1 + id_len]
                    for k, aid in enumerate(id_ava):
                        if aid not in doc_ids:
                            doc_ids.append(aid)
                            doc_scores.append(res_score)

            for index, doc_id in enumerate(doc_ids):
                res_doc = vs.docstore.search(doc_id)
                results.append((Document(page_content=res_doc.page_content, metadata=res_doc.metadata), doc_scores[index]))
        return results

    def do_add_doc(self,
                   docs: List[Document],
                   **kwargs,
                   ) -> List[Dict]:
        data = self._docs_to_embeddings(docs) # 将向量化单独出来可以减少向量库的锁定时间

        with self.load_vector_store().acquire() as vs:
            ids = vs.add_embeddings(text_embeddings=zip(data["texts"], data["embeddings"]),
                                    metadatas=data["metadatas"],
                                    ids=kwargs.get("ids"))
            if not kwargs.get("not_refresh_vs_cache"):
                vs.save_local(self.vs_path)
        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        torch_gc()
        return doc_infos

    def do_delete_doc(self,
                      kb_file: KnowledgeFile,
                      **kwargs):
        with self.load_vector_store().acquire() as vs:
            ids = [k for k, v in vs.docstore._dict.items() if v.metadata.get("source").lower() == kb_file.filename.lower()]
            if len(ids) > 0:
                vs.delete(ids)
            if not kwargs.get("not_refresh_vs_cache"):
                vs.save_local(self.vs_path)
        return ids

    def do_clear_vs(self):
        with kb_faiss_pool.atomic:
            kb_faiss_pool.pop((self.kb_name, self.vector_name))
        try:
            shutil.rmtree(self.vs_path)
        except Exception:
            ...
        os.makedirs(self.vs_path, exist_ok=True)

    def exist_doc(self, file_name: str):
        if super().exist_doc(file_name):
            return "in_db"

        content_path = os.path.join(self.kb_path, "content")
        if os.path.isfile(os.path.join(content_path, file_name)):
            return "in_folder"
        else:
            return False


if __name__ == '__main__':
    # faissService = FaissKBService("1951932460")
    # faissService.add_doc(KnowledgeFile("README.md", "test"))
    # faissService.delete_doc(KnowledgeFile("README.md", "test"))
    # faissService.do_drop_kb()
    # print(faissService.search_docs("内部事件定义"))

    faissService = FaissKBService("test")
    faissService.add_doc(KnowledgeFile("README.md", "test"))
    print(faissService.search_docs("Langchain"))
