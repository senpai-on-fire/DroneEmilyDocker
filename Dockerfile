# Use Python 3.7 because TF 1.x requires <=3.7
FROM python:3.9.21

# System basics (optional: add build tools if you compile anything)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Prevent python from buffering stdout; helpful for seeing logs immediately
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Workdir inside container
WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy your project (Emily_Drone.py and anything it imports)
# If you will bind-mount your code at runtime, you can skip this COPY
COPY . /app

# Default command: run your script then print the file
# Adjust path if your script writes coefficients.csv elsewhere
CMD bash -lc "python3 Emily_Drone.py --model ltc && \
              echo -e '\n----- coefficients.csv -----\n' && \
              cat coefficients.csv"

