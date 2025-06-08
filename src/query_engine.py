def get_query_engine(index):
    return index.as_query_engine()

def query_index(query_engine, query_text):
    return query_engine.query(query_text)
