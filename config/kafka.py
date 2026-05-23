from kafka import KafkaProducer, KafkaConsumer
from .settings import settings

kafka_client = {
    "producer": KafkaProducer(bootstrap_servers=settings.kafka_bootstrap_servers),
    "consumer": KafkaConsumer(bootstrap_servers=settings.kafka_bootstrap_servers),
}
