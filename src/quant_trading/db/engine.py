from sqlalchemy import create_engine


def make_engine(mysql_dsn: str):
    return create_engine(mysql_dsn, pool_pre_ping=True)
