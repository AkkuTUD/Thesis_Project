FROM registry.roqs.basf.net/base-images/python:latest

# /app has already been defined and set as workdir in the base-image
ADD requirements.txt /app
RUN pip3 install -U -r requirements.txt && \
    rm requirements.txt
EXPOSE 5000
USER app
COPY myProxyFix.py myProxyFix.py
ADD . /app
CMD ["gunicorn", "-w", "10", "-b", "0.0.0.0:5000", "-t", "100000", "--max-requests", "20", "--access-logfile", "-", "--error-logfile", "-", "--capture-output", "server:app"]