#!/bin/bash

#if [ ! -d "test" ]
#then
#  mkdir test
#fi
pip3 install pip-tools
#
apt-get update -y
apt-get install -y libmariadb-dev
apt-get install -y mysql-client

DATABASE_TO_COPY_INTO="baseball"
DATABASE_FILE="baseball.sql"

if mysql -h mariadb -u root -ptest -e "use ${DATABASE_TO_COPY_INTO}" >/dev/null 2>&1; then # pragma: allowlist secret
    echo "Database already exists"
else
    echo "Database does not exist, creating..."
    mysql -h mariadb -u root -ptest -e "CREATE DATABASE ${DATABASE_TO_COPY_INTO}" # pragma: allowlist secret
    mysql -h mariadb -u root -ptest ${DATABASE_TO_COPY_INTO} < ${DATABASE_FILE} # pragma: allowlist secret
fi

pip3 install -r requirements.dev.txt
pip3 install -r requirements.txt

# python3 dockertest.py

python3 final.py