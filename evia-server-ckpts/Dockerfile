FROM python:latest

WORKDIR /opt/webscrapper/KleinanzeigenWebscrapper/

COPY . .

RUN pip install bs4

RUN pip install telepot

RUN pip install mariadb

RUN pip install requests

CMD ["python", "main.py"]
