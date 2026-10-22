"""Script to generate realistic academic sample PDFs for EduRAG demonstration and testing."""

import os
import pymupdf as fitz


def create_cs_sample_pdf(output_path: str):
    """Generate a 5-page Computer Science & Cryptography academic lecture PDF."""
    doc = fitz.open()

    pages_content = [
        # Page 1
        (
            "Chapter 1: Algorithm Analysis and Asymptotic Notation",
            [
                "Asymptotic analysis evaluates the efficiency of algorithms as the input size n grows toward infinity.",
                "Big-O notation represents an asymptotic upper bound. Formally, f(n) = O(g(n)) if there exist positive constants c and n0 such that 0 <= f(n) <= c * g(n) for all n >= n0.",
                "Common complexity classes include O(1) constant time, O(log n) logarithmic time, O(n) linear time, O(n log n) linearithmic time, and O(n^2) quadratic time.",
                "In computer science education, understanding asymptotic behavior is fundamental for selecting optimal data structures such as balanced binary search trees and hash tables.",
            ]
        ),
        # Page 2
        (
            "Chapter 2: Sorting Algorithms and Divide-and-Conquer",
            [
                "Divide-and-conquer is an algorithmic paradigm that recursively breaks a problem into smaller subproblems of the same type until they become simple enough to solve directly.",
                "MergeSort is an asymptotically optimal comparison-based sorting algorithm. Its recurrence relation is T(n) = 2T(n/2) + O(n), yielding a guaranteed worst-case time complexity of O(n log n).",
                "QuickSort selects a pivot element and partitions the array into elements less than and greater than the pivot. Its average-case time complexity is O(n log n), while its worst-case complexity is O(n^2) when poor pivots are chosen.",
                "In terms of space complexity, MergeSort requires O(n) auxiliary space for merging subarrays, whereas in-place QuickSort requires O(log n) space for recursive call stack frames.",
            ]
        ),
        # Page 3
        (
            "Chapter 3: Graph Algorithms and Shortest Paths",
            [
                "A graph G = (V, E) consists of a set of vertices V and edges E connecting pairs of vertices.",
                "Dijkstra's algorithm computes the single-source shortest path for a directed or undirected graph with non-negative edge weights.",
                "Using a min-priority heap, Dijkstra's algorithm achieves a time complexity of O((V + E) log V). If a Fibonacci heap is utilized, the running time decreases to O(E + V log V).",
                "The Bellman-Ford algorithm can handle graphs containing negative edge weights and is capable of detecting negative weight cycles in O(V * E) time.",
            ]
        ),
        # Page 4
        (
            "Chapter 4: Cryptographic Foundations and RSA Algorithm",
            [
                "Asymmetric cryptography, or public-key cryptography, utilizes mathematically linked key pairs consisting of a public key and a private key.",
                "The RSA algorithm was invented in 1977 by Ron Rivest, Adi Shamir, and Leonard Adleman. Its security relies on the computational difficulty of factoring the product of two large prime numbers.",
                "Key generation involves choosing two distinct prime numbers p and q, computing the modulus n = p * q, and Euler's totient function phi(n) = (p - 1) * (q - 1).",
                "A public exponent e is chosen such that 1 < e < phi(n) and gcd(e, phi(n)) = 1. The private decryption key d satisfies the congruence d * e = 1 (mod phi(n)).",
            ]
        ),
        # Page 5
        (
            "Chapter 5: Dynamic Programming and Optimal Substructure",
            [
                "Dynamic programming is an algorithmic technique for solving optimization problems by decomposing them into overlapping subproblems and optimal substructures.",
                "The two primary implementations of dynamic programming are top-down memoization (recursive caching) and bottom-up tabulation (iterative table construction).",
                "Classic examples include the 0/1 Knapsack problem, Longest Common Subsequence (LCS), and Matrix Chain Multiplication.",
                "By storing the solutions to subproblems rather than recomputing them exponentially, dynamic programming typically reduces polynomial time complexities from O(2^n) to O(n^2) or O(n * W).",
            ]
        ),
    ]

    for title, paragraphs in pages_content:
        page = doc.new_page(width=595, height=842)  # A4 standard

        # Page Header
        page.insert_text(fitz.Point(50, 50), "EduRAG Academic Lecture Notes Series — Computer Science", fontsize=9, color=(0.3, 0.3, 0.5))
        page.draw_line(fitz.Point(50, 60), fitz.Point(545, 60), color=(0.7, 0.7, 0.7), width=1)

        # Chapter Title
        page.insert_text(fitz.Point(50, 95), title, fontsize=16, fontname="helv", color=(0.1, 0.2, 0.4))

        # Paragraphs
        y = 130
        for para in paragraphs:
            rect = fitz.Rect(50, y, 545, y + 100)
            page.insert_textbox(rect, para, fontsize=11, fontname="helv", color=(0.15, 0.15, 0.15), align=0)
            y += 110

        # Page Footer
        page.draw_line(fitz.Point(50, 800), fitz.Point(545, 800), color=(0.7, 0.7, 0.7), width=1)
        page.insert_text(fitz.Point(50, 815), "Open Educational Resource (CC-BY-4.0)", fontsize=8, color=(0.4, 0.4, 0.4))
        page.insert_text(fitz.Point(480, 815), f"Page {len(doc)}", fontsize=9, color=(0.2, 0.2, 0.2))

    doc.save(output_path)
    doc.close()
    print(f"Created sample PDF at: {output_path}")


def create_physics_sample_pdf(output_path: str):
    """Generate a 4-page Physics & Quantum Mechanics sample lecture PDF."""
    doc = fitz.open()

    pages_content = [
        # Page 1
        (
            "Lecture 1: Foundations of Quantum Physics",
            [
                "Quantum mechanics arose in the early twentieth century to describe phenomena at microscopic scales where classical Newtonian mechanics fails.",
                "Wave-particle duality states that matter and light exhibit behaviors characteristic of both waves and discrete particles.",
                "Max Planck solved the ultraviolet catastrophe in black-body radiation by postulating that electromagnetic energy is quantized in discrete packets called quanta: E = h * nu.",
                "Albert Einstein explained the photoelectric effect in 1905 by demonstrating that light consists of localized photons carrying energy proportional to frequency.",
            ]
        ),
        # Page 2
        (
            "Lecture 2: The Schrödinger Wave Equation",
            [
                "The state of a quantum physical system is completely described by its complex wave function Psi(r, t).",
                "The time-dependent Schrödinger equation describes how the quantum state of a non-relativistic physical system changes over time: i * h_bar * d(Psi)/dt = H * Psi, where H is the Hamiltonian operator.",
                "Born's statistical interpretation states that the square of the absolute value of the wave function, |Psi(r, t)|^2, gives the probability density of finding the particle at position r at time t.",
                "Wave functions must satisfy mathematical boundary conditions including square-integrability, continuity, and single-valuedness across physical space.",
            ]
        ),
        # Page 3
        (
            "Lecture 3: Heisenberg Uncertainty Principle",
            [
                "Formulated by Werner Heisenberg in 1927, the uncertainty principle establishes a fundamental limit to the precision with which certain pairs of physical properties can be known simultaneously.",
                "The position-momentum uncertainty relation is Delta(x) * Delta(p) >= h_bar / 2, indicating that position and linear momentum are canonically conjugate observables.",
                "This principle is an intrinsic wave property of quantum systems, not merely a limitation of experimental measuring instruments.",
                "Similarly, the energy-time uncertainty relation Delta(E) * Delta(t) >= h_bar / 2 governs the lifetime of excited quantum states and particle resonance widths.",
            ]
        ),
        # Page 4
        (
            "Lecture 4: Quantum Entanglement and Non-Locality",
            [
                "Quantum entanglement occurs when two or more particles become interconnected such that the quantum state of each particle cannot be described independently of the state of the others.",
                "In 1935, Albert Einstein, Boris Podolsky, and Nathan Rosen proposed the EPR paradox, questioning whether quantum mechanics is a complete theory or whether hidden variables exist.",
                "In 1964, John Stewart Bell formulated Bell's theorem and Bell inequalities, proving that no local hidden-variable theory can reproduce all predictions of quantum mechanics.",
                "Experimental verifications of Bell inequality violations have laid the foundation for modern quantum computing, quantum key distribution (QKD), and quantum teleportation protocols.",
            ]
        ),
    ]

    for title, paragraphs in pages_content:
        page = doc.new_page(width=595, height=842)

        page.insert_text(fitz.Point(50, 50), "EduRAG Academic Lecture Notes Series — Modern Physics", fontsize=9, color=(0.3, 0.4, 0.3))
        page.draw_line(fitz.Point(50, 60), fitz.Point(545, 60), color=(0.7, 0.7, 0.7), width=1)

        page.insert_text(fitz.Point(50, 95), title, fontsize=16, fontname="helv", color=(0.1, 0.3, 0.2))

        y = 130
        for para in paragraphs:
            rect = fitz.Rect(50, y, 545, y + 100)
            page.insert_textbox(rect, para, fontsize=11, fontname="helv", color=(0.15, 0.15, 0.15), align=0)
            y += 110

        page.draw_line(fitz.Point(50, 800), fitz.Point(545, 800), color=(0.7, 0.7, 0.7), width=1)
        page.insert_text(fitz.Point(50, 815), "Open Educational Resource (CC-BY-4.0)", fontsize=8, color=(0.4, 0.4, 0.4))
        page.insert_text(fitz.Point(480, 815), f"Page {len(doc)}", fontsize=9, color=(0.2, 0.2, 0.2))

    doc.save(output_path)
    doc.close()
    print(f"Created sample PDF at: {output_path}")


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    cs_path = os.path.join(out_dir, "computer_science_algorithms.pdf")
    physics_path = os.path.join(out_dir, "physics_quantum_mechanics.pdf")

    create_cs_sample_pdf(cs_path)
    create_physics_sample_pdf(physics_path)
