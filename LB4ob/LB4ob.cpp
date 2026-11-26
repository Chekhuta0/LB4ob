#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <windows.h>
#include <mpi.h>
using namespace std;

double poly(double a, double b) {
    return (2.8 * a * a - 0.8) * b + (4.4 * b * b - 0.9) * a + 0.25;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    SetConsoleOutputCP(1251);

    int n = 0;
    if (rank == 0) {
        while (true) {
            cout << "N (1000..1000000): ";
            if (cin >> n && n >= 1000 && n <= 1000000) break;
            cin.clear();
            cin.ignore(10000, '\n');
            cout << "Помилка: введіть число від 10000 до 1000000\n";
        }
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<double> A, B, C;
    double tSeq = 0, sumSeq = 0;

    if (rank == 0) {
        A.resize(n); B.resize(n); C.resize(n);
        mt19937 gen(random_device{}());
        uniform_real_distribution<double> rnd(-1, 1);
        for (int i = 0; i < n; i++) { A[i] = rnd(gen); B[i] = rnd(gen); }

        double t = MPI_Wtime();
        for (int r = 0; r < 100; r++)
            for (int i = 0; i < n; i++)
                C[i] = poly(A[i], B[i]);
        tSeq = (MPI_Wtime() - t) / 100;
        for (int i = 0; i < n; i++) sumSeq += C[i];

        cout << fixed << setprecision(6) << "\nПерші елементи (i, A[i], B[i], C[i]):\n";
        for (int i = 0; i < 5; i++)
            cout << i << ": " << A[i] << "  " << B[i] << "  " << C[i] << "\n";
        cout << "Контрольна сума (послідовно): " << sumSeq << "\n\n";
    }

    vector<int> cnt(p), dsp(p);
    for (int q = 0, off = 0; q < p; q++) {
        cnt[q] = n / p + (q < n % p ? 1 : 0);
        dsp[q] = off;
        off += cnt[q];
    }
    vector<double> la(cnt[rank]), lb(cnt[rank]), lc(cnt[rank]);

    MPI_Barrier(MPI_COMM_WORLD);
    double t = MPI_Wtime();
    for (int r = 0; r < 100; r++) {
        MPI_Scatterv(A.data(), cnt.data(), dsp.data(), MPI_DOUBLE, la.data(), cnt[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(B.data(), cnt.data(), dsp.data(), MPI_DOUBLE, lb.data(), cnt[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for (int i = 0; i < cnt[rank]; i++) lc[i] = poly(la[i], lb[i]);
        MPI_Gatherv(lc.data(), cnt[rank], MPI_DOUBLE, C.data(), cnt.data(), dsp.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    double mine = (MPI_Wtime() - t) / 100, tPar = 0;
    MPI_Reduce(&mine, &tPar, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double sumPar = 0;
        for (int i = 0; i < n; i++) sumPar += C[i];
        cout << fixed << setprecision(4)
            << "Послідовно: " << tSeq * 1000 << " мс\n"
            << "Паралельно (" << p << " проц.): " << tPar * 1000 << " мс, прискорення x" << tSeq / tPar
            << ", контрольна сума: " << setprecision(6) << sumPar
            << (sumPar == sumSeq ? " (збігається)\n" : " (НЕ ЗБІГАЄТЬСЯ!)\n");
    }

    MPI_Finalize();
    return 0;
}