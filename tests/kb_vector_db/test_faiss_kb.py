from server.knowledge_base.kb_service.base import KBService, KBServiceFactory
from server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
from server.knowledge_base.migrate import create_tables
from server.knowledge_base.utils import KnowledgeFile


# kbService = FaissKBService("test")
# test_kb_name = "test"
# test_file_name = "README.md"
# testKnowledgeFile = KnowledgeFile(test_file_name, test_kb_name)
# search_content = "Langchain-Chatchat"
# kbService = FaissKBService("1951932460")
kbService = KBServiceFactory.get_service_by_name("1951932460")
if kbService is None:
    kbService = KBServiceFactory.get_service("1951932460", "faiss")
test_kb_name = "1951932460"
test_file_name = "/1951932460/IT内部事件管理细则（试行）.pdf"
testKnowledgeFile = KnowledgeFile(test_file_name, test_kb_name, from_minio=True)
search_content = "内部事件定义"


def test_init():
    create_tables()


def test_create_db():
    assert kbService.create_kb()


def test_add_doc():
    assert kbService.add_doc(testKnowledgeFile)


def test_search_db():
    result = kbService.search_docs(search_content)
    for item in result:
        print(item)
    assert len(result) > 0


def test_delete_doc():
    assert kbService.delete_doc(testKnowledgeFile)


def test_delete_db():
    assert kbService.drop_kb()
