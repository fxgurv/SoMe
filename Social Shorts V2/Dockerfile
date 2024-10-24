FROM nvidia/cuda:12.3.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Paris

# Remove any third-party apt sources and install basic utilities
RUN rm -f /etc/apt/sources.list.d/*.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    sudo \
    git \
    wget \
    procps \
    git-lfs \
    zip \
    unzip \
    htop \
    vim \
    nano \
    bzip2 \
    libx11-6 \
    build-essential \
    libsndfile-dev \
    software-properties-common \
    xvfb \
    imagemagick \
    gpg \
    gpg-agent \
    gnupg \
    apt-transport-https \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libnss3 \
    libgconf-2-4 \
    libasound2 \
    libfontconfig1 \
    libxcb1 \
    libxcb-shm0 \
    libxcb-render0 \
    libxcb-xinerama0 \
    libtinfo5 \
    libasound2 \
    libpulse0 \
    libegl1 \
    libpci3 \
    libgl1 \
    libopengl0 \
    libegl1-mesa \
    libopus0 \
    libopus-dev \
    net-tools \
    iputils-ping \
    iproute2 \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

RUN sed -i '/<policy domain="path" rights="none" pattern="@\*"/d' /etc/ImageMagick-6/policy.xml

# Install nvtop
RUN add-apt-repository ppa:flexiondotorg/nvtop && \
    apt-get update && \
    apt-get install -y --no-install-recommends nvtop && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g configurable-http-proxy

# Install latest Firefox
RUN sudo install -d -m 0755 /etc/apt/keyrings && \
    wget -q https://packages.mozilla.org/apt/repo-signing-key.gpg -O- | sudo tee /etc/apt/keyrings/packages.mozilla.org.asc > /dev/null && \
    echo "deb [signed-by=/etc/apt/keyrings/packages.mozilla.org.asc] https://packages.mozilla.org/apt mozilla main" | sudo tee -a /etc/apt/sources.list.d/mozilla.list > /dev/null && \
    echo 'Package: *\nPin: origin packages.mozilla.org\nPin-Priority: 1000' | sudo tee /etc/apt/preferences.d/mozilla && \
    sudo apt-get update && sudo apt-get install -y firefox

# Download and install latest GeckoDriver for Firefox
RUN GECKO_LATEST=$(curl -sL https://api.github.com/repos/mozilla/geckodriver/releases/latest | grep tag_name | cut -d '"' -f 4) && \
    wget https://github.com/mozilla/geckodriver/releases/download/${GECKO_LATEST}/geckodriver-${GECKO_LATEST}-linux64.tar.gz && \
    tar -xvzf geckodriver-${GECKO_LATEST}-linux64.tar.gz && \
    sudo mv geckodriver /usr/local/bin/ && \
    sudo chmod +x /usr/local/bin/geckodriver && \
    rm geckodriver-${GECKO_LATEST}-linux64.tar.gz && \
    geckodriver --version

# Install Google Chrome
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | sudo apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list && \
    sudo apt-get update && \
    sudo apt-get install -y google-chrome-stable

# Install ChromeDriver
RUN CHROME_DRIVER_VERSION=$(curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE) && \
    wget -q -O /tmp/chromedriver.zip https://chromedriver.storage.googleapis.com/${CHROME_DRIVER_VERSION}/chromedriver_linux64.zip && \
    unzip /tmp/chromedriver.zip -d /usr/local/bin/ && \
    rm /tmp/chromedriver.zip && \
    chmod +x /usr/local/bin/chromedriver

# Install Microsoft Edge
RUN curl https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.gpg && \
    sudo install -o root -g root -m 644 microsoft.gpg /etc/apt/trusted.gpg.d/ && \
    sudo sh -c 'echo "deb [arch=amd64] https://packages.microsoft.com/repos/edge stable main" > /etc/apt/sources.list.d/microsoft-edge-dev.list' && \
    sudo rm microsoft.gpg && \
    sudo apt-get update && \
    sudo apt-get install -y microsoft-edge-stable

# Install Edge WebDriver
RUN EDGE_DRIVER_VERSION=$(curl -s https://msedgedriver.azureedge.net/LATEST_STABLE) && \
    wget -q -O /tmp/edgedriver.zip https://msedgedriver.azureedge.net/${EDGE_DRIVER_VERSION}/edgedriver_linux64.zip && \
    unzip /tmp/edgedriver.zip -d /usr/local/bin/ && \
    rm /tmp/edgedriver.zip && \
    mv /usr/local/bin/msedgedriver /usr/local/bin/edgedriver && \
    chmod +x /usr/local/bin/edgedriver


# Create a working directory
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN mkdir $HOME/.cache $HOME/.config \
 && chmod -R 777 $HOME

# Set up the Conda environment
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=$HOME/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda clean -ya

WORKDIR $HOME/app

#######################################
# Start root user section
#######################################

USER root

# User Debian packages
## Security warning : Potential user code executed as root (build time)
RUN --mount=target=/root/packages.txt,source=packages.txt \
    apt-get update && \
    xargs -r -a /root/packages.txt apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=target=/root/on_startup.sh,source=on_startup.sh,readwrite \
    bash /root/on_startup.sh

RUN mkdir /data && chown user:user /data

#######################################
# End root user section
#######################################

USER user

# Install Jupyter Lab
RUN conda install -y -c conda-forge jupyterlab

# Python packages
RUN --mount=target=requirements.txt,source=requirements.txt \
    pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

RUN chmod +x start_server.sh

COPY --chown=user login.html /home/user/miniconda/lib/python3.9/site-packages/jupyter_server/templates/login.html

ENV PYTHONUNBUFFERED=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface \
    SYSTEM=spaces \
    SHELL=/bin/bash

CMD ["./start_server.sh"]
