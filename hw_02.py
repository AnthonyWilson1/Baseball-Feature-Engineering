import sys

import sqlalchemy


def main():

    db_user = "root"
    db_pass = "test"  # pragma: allowlist secret
    db_host = "localhost"
    db_database = "baseball"
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )
    sql_engine = sqlalchemy.create_engine(connect_string)
    print(sql_engine)


if __name__ == "__main__":
    sys.exit(main())
