from airflow.utils.log.logging_mixin import LoggingMixin
from logging import Logger

class OperatorLogger(object):

    def getLogger() -> Logger:
        return LoggingMixin().log