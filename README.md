# MLOps Course - Student Machine Setup

Use this guide to prepare a basic development environment for the first MLOps class.

The first class will focus on working with Jupyter notebooks and converting them into Python scripts, version-controlling your work, and running code reliably in a virtual environment.

## 0) Supported operating systems

- Windows 10/11: use WSL2 (Ubuntu recommended)
- macOS (Apple Silicon or Intel)
- Linux (Ubuntu/Debian-based)

Instructions below include platform-specific steps where needed.

---

## 1) Windows only — Install WSL2 (Ubuntu)

1. Open PowerShell as Administrator and run:
   ```powershell
   wsl --install -d Ubuntu
   ```
   - If prompted, restart your computer.
2. Launch "Ubuntu" from Start Menu, create a UNIX username and password.
3. Update packages inside Ubuntu:
   ```bash
   sudo apt update && sudo apt -y upgrade
   ```

Why WSL2? It provides a Linux environment that matches most production tooling and simplifies Python setup.

---

## 2) Install VS Code

1. Download and install VS Code from the official site: [code.visualstudio.com](https://code.visualstudio.com)
2. Recommended extensions:
   - "Python" (Microsoft)
   - "Jupyter" (Microsoft)
   - "Remote - WSL" (Windows only)

On Windows, after installing "Remote - WSL", open VS Code and connect to Ubuntu: press F1 → "WSL: Connect to WSL".

---

## 3) Install Git

- Windows (inside WSL Ubuntu):
  ```bash
  sudo apt update && sudo apt -y install git
  ```
  (Optional) Install Git for Windows as well: [git-scm.com](https://git-scm.com)

- macOS:
  - If prompted, install Command Line Tools, or install via Homebrew:
    ```bash
    brew install git
    ```

- Linux (Debian/Ubuntu):
  ```bash
  sudo apt update && sudo apt -y install git
  ```

Configure your identity (run in your primary dev shell: WSL Ubuntu/macOS/Linux):
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

(Optional but recommended) Setup SSH for GitHub: follow GitHub docs: [docs.github.com → Generate a new SSH key and add it to GitHub](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

---

## 4) Create a GitHub account

1. Sign up at: [github.com](https://github.com)
2. Verify your email.
3. (Optional) Enable 2FA for security.

---

## 5) Install Python 3.12 with pyenv

We will use Python 3.12 and manage it with `pyenv`. This keeps your system Python untouched and makes switching versions easy.

### 5.1 Install pyenv

- Windows (WSL Ubuntu):
  ```bash
  # Dependencies
  sudo apt update && sudo apt -y install build-essential curl git zlib1g-dev \
    libssl-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev \
    liblzma-dev tk-dev

  # Install pyenv (via installer)
  curl https://pyenv.run | bash

  # Shell init (Zsh)
  echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
  echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
  echo 'eval "$(pyenv init -)"' >> ~/.zshrc
  exec zsh -l
  ```

- macOS (with Homebrew):
  ```bash
  brew update
  brew install pyenv
  # Shell init (Zsh)
  echo 'eval "$(pyenv init -)"' >> ~/.zshrc
  exec zsh -l
  ```

- Linux (Ubuntu/Debian):
  ```bash
  sudo apt update && sudo apt -y install build-essential curl git zlib1g-dev \
    libssl-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev \
    liblzma-dev tk-dev
  curl https://pyenv.run | bash
  echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
  echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
  echo 'eval "$(pyenv init -)"' >> ~/.bashrc
  exec bash -l
  ```

Verify pyenv is installed:
```bash
pyenv --version
```

### 5.2 Install Python 3.12 and set local version

```bash
pyenv install 3.12.6   # or the latest 3.12.x shown by: pyenv install -l | grep " 3.12"
pyenv global 3.12.6     # or use `pyenv local 3.12.6` inside your course folder
python --version        # should show Python 3.12.x
pip --version
```

---

## 6) Install Docker

Docker is essential for containerizing applications and ensuring consistent environments across different systems. We'll install Docker Desktop for Windows/macOS or Docker Engine for Linux.

### 6.1 Windows (WSL2)

Docker Desktop for Windows integrates seamlessly with WSL2.

1. **Download Docker Desktop**:
   - Visit [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
   - Download "Docker Desktop for Windows"
   - Run the installer (`Docker Desktop Installer.exe`)

2. **Installation Steps**:
   - When prompted, ensure "Use WSL 2 instead of Hyper-V" is checked
   - Follow the installation wizard
   - Restart your computer when prompted

3. **Launch Docker Desktop**:
   - Start Docker Desktop from the Start Menu
   - Wait for Docker to start (you'll see a Docker icon in the system tray)

4. **Verify Installation** (inside WSL Ubuntu):
   ```bash
   docker --version
   docker-compose --version
   docker run hello-world
   ```
   The `hello-world` command should pull and run a test container successfully.

5. **Optional: Configure WSL2 Integration**:
   - Open Docker Desktop → Settings → Resources → WSL Integration
   - Enable integration with your Ubuntu distribution
   - Click "Apply & Restart"

### 6.2 macOS

Docker Desktop for macOS supports both Apple Silicon (M1/M2/M3) and Intel processors.

#### For Apple Silicon (M1/M2/M3) Macs:

1. **Download Docker Desktop**:
   - Visit [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
   - Download "Docker Desktop for Mac with Apple Silicon"
   - Open the downloaded `.dmg` file

2. **Installation**:
   - Drag Docker.app to Applications folder
   - Open Docker from Applications (you may need to authorize in System Preferences → Security & Privacy)
   - Enter your macOS password when prompted

3. **Verify Installation**:
   ```bash
   docker --version
   docker-compose --version
   docker run hello-world
   ```

#### For Intel Macs:

1. **Download Docker Desktop**:
   - Visit [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
   - Download "Docker Desktop for Mac with Intel chip"
   - Follow the same installation steps as above

**Alternative (via Homebrew)**:
```bash
# Install via Homebrew (works for both Apple Silicon and Intel)
brew install --cask docker

# Start Docker Desktop
open /Applications/Docker.app

# Wait for Docker to start, then verify
docker --version
docker-compose --version
docker run hello-world
```

### 6.3 Linux (Ubuntu/Debian-based)

We'll install Docker Engine using the official Docker repository (recommended method).

1. **Remove old versions** (if any):
   ```bash
   sudo apt-get remove docker docker-engine docker.io containerd runc
   ```

2. **Install prerequisites**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y \
       ca-certificates \
       curl \
       gnupg \
       lsb-release
   ```

3. **Add Docker's official GPG key**:
   ```bash
   sudo mkdir -p /etc/apt/keyrings
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
   ```

4. **Set up Docker repository**:
   ```bash
   echo \
     "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
     $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
   ```

5. **Install Docker Engine, CLI, and Containerd**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

6. **Add your user to the docker group** (to run Docker without sudo):
   ```bash
   sudo usermod -aG docker $USER
   ```
   **Important**: Log out and log back in (or restart your terminal) for this change to take effect.

7. **Verify Installation**:
   ```bash
   docker --version
   docker-compose --version
   docker run hello-world
   ```

### 6.4 Verify Docker Installation (All Platforms)

After installation, verify that Docker is working correctly:

```bash
# Check Docker version
docker --version
# Should output something like: Docker version 24.0.7, build afdd53b

# Check Docker Compose version
docker-compose --version
# Should output something like: Docker Compose version v2.23.0

# Run a test container
docker run hello-world
# This should download and run a test image, printing "Hello from Docker!"

# Check Docker daemon is running
docker info
# This should show detailed system information
```

### Troubleshooting

**Windows WSL2**:
- If `docker` command is not found in WSL, ensure Docker Desktop is running and WSL integration is enabled in Docker Desktop settings.
- If you get permission errors, make sure your WSL Ubuntu user is in the `docker` group (usually handled automatically by Docker Desktop).

**macOS**:
- If Docker Desktop doesn't start, check System Preferences → Security & Privacy → General for authorization prompts.
- On Apple Silicon, ensure you're using the Apple Silicon version of Docker Desktop.

**Linux**:
- If you get "permission denied" errors, ensure you've added your user to the docker group and logged out/in.
- If Docker daemon isn't running, start it with: `sudo systemctl start docker`
- Enable Docker to start on boot: `sudo systemctl enable docker`

**General**:
- If `hello-world` fails to pull, check your internet connection and Docker daemon status.
- For more help, visit: [docs.docker.com/get-docker](https://docs.docker.com/get-docker/)

---

## 7) Verify Git + GitHub access

1. In your dev shell, generate an SSH key if you chose SSH:
   ```bash
   ssh-keygen -t ed25519 -C "your.email@example.com"
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ```
2. Add the public key (`~/.ssh/id_ed25519.pub`) to GitHub → Settings → SSH and GPG keys.
3. Test:
   ```bash
   ssh -T git@github.com
   ```
   You should see a success message.

---

## FAQ: Do we need other tools for the first class?

For the first session (Jupyter notebook to Python script), keep it minimal:

- Required:
  - Python 3.12 + `venv`
  - VS Code with Python + Jupyter extensions
  - Jupyter/JupyterLab + ipykernel
  - Git + GitHub account


We will introduce additional tooling (linting, testing, packaging, environment management, containers, CI) in later sessions. For now, please ensure the steps above are completed and that you can run a Jupyter notebook and a simple Python script inside your 3.12 virtual environment.

