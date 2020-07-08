FROM python:3

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /usr/src/MB-FTE
COPY . .
RUN pip install --no-cache-dir -r deps.txt

CMD ["./run.sh"]
