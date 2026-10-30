# Contributing to Drivox

Thank you for your interest in contributing to **Drivox**! We welcome contributions to improve algorithms, optimize edge inference, expand test coverage, and enhance fleet telemetry capabilities.

---

## 🛠️ Development Setup

1. **Clone the repository**:
   ```bash
   git clone git@github.com:aditighoshagd/Drivox.git
   cd Drivox
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run automated tests**:
   ```bash
   python main.py --mode test
   ```

---

## 📋 Guidelines

- **Code Quality**: Follow PEP 8 style guidelines. Ensure type hints and docstrings are provided for public interfaces.
- **Testing**: Add unit and integration tests under `tests/` for any new mathematical or vision feature.
- **Edge Performance**: Ensure changes do not degrade the 30+ FPS edge latency budget on CPU hardware.
- **Pull Requests**: Submit clear pull requests with detailed descriptions of changes and benchmark metrics.

---

## 📜 License
By contributing to DriverPulse AI, you agree that your contributions will be licensed under the MIT License.
