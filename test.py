from elasticsearch import Elasticsearch

es = Elasticsearch(
    ['https://localhost:9200'],
    verify_certs=True,
    ca_certs='path_to_http_ca.crt',
    client_cert='path_to_client_cert.pem',
    client_key='path_to_client_key.pem'
)
