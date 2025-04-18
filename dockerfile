FROM python:3.11

WORKDIR /app

COPY . /app
COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt
RUN pip install uvicorn
RUN pip install openpyxl

RUN which uvicorn

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

