FROM python:3

COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt --no-cache
WORKDIR /www/app/

COPY app.py .
CMD ["streamlit", "run", "app.py"]