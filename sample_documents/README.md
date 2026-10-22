# Educational Document Datasets for EduRAG

This directory contains information on how to obtain, upload, and test educational textbooks and lecture notes with **EduRAG**.

---

## 1. Included Sample Educational PDFs

For rapid evaluation, offline testing, and live hackathon demonstrations, two authentic educational PDFs are provided:

1. **`computer_science_algorithms.pdf`** (5 pages)
   - **Page 1**: Algorithm Analysis and Asymptotic Big-O Notation
   - **Page 2**: Sorting Algorithms, Recurrence Relations, and MergeSort vs QuickSort
   - **Page 3**: Graph Algorithms (Dijkstra's Shortest Path & Bellman-Ford)
   - **Page 4**: Cryptographic Foundations and the RSA Algorithm (Public/Private keys, Euler's totient)
   - **Page 5**: Dynamic Programming, Memoization vs Tabulation, and 0/1 Knapsack

2. **`physics_quantum_mechanics.pdf`** (4 pages)
   - **Page 1**: Foundations of Quantum Physics, Planck's quanta, and Photoelectric Effect
   - **Page 2**: The Schrödinger Wave Equation and Born's Statistical Probability Interpretation
   - **Page 3**: Heisenberg Uncertainty Principle ($\Delta x \cdot \Delta p \ge \hbar/2$)
   - **Page 4**: Quantum Entanglement, EPR Paradox, and Bell's Theorem

---

## 2. Public Academic Document Repositories

Students and educators can use EduRAG with any standard educational PDF from open repositories:

### OpenStax Textbooks (Recommended)
- **URL**: [https://openstax.org/](https://openstax.org/)
- **License**: Creative Commons Attribution (CC-BY)
- **Available Topics**: University Physics, Calculus, Biology, Chemistry, Computer Applications, Economics, Psychology, and US History.
- **How to download**:
  1. Visit [openstax.org/subjects](https://openstax.org/subjects).
  2. Select any subject (e.g., *College Physics* or *Calculus Volume 1*).
  3. Click **"Get the Book"** and choose **"Download a PDF"**.
  4. Upload the PDF directly into EduRAG via the sidebar.

### MIT OpenCourseWare (OCW)
- **URL**: [https://ocw.mit.edu/](https://ocw.mit.edu/)
- **License**: CC-BY-NC-SA
- **Available Topics**: Undergraduate and graduate lecture notes, exam problem sets, and reading materials from MIT courses.
- **How to download**:
  1. Browse courses by department (e.g., *Electrical Engineering & Computer Science*, *Physics*).
  2. Navigate to the **"Lecture Notes"** or **"Readings"** section.
  3. Download individual topic PDFs and upload them to EduRAG.

---

## 3. Supported File Types & Best Practices

- **Supported Formats**: Standard digital PDF files with selectable text.
- **Scanned Documents**: Scanned image-only PDFs will be detected by EduRAG and flagged with an informative message explaining that Optical Character Recognition (OCR) is required.
- **Page Numbering**: EduRAG indexes PDFs strictly using physical **1-based PDF page numbers**, ensuring students can instantly jump to the exact page in their PDF viewer.
