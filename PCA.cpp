/*********************************************************************************
Author: João Fontes Gonçalves
To compile: g++ -I /home/dell/eigen-3.3.7/ PCA.cpp -o PCA
*********************************************************************************/

#include <bits/stdc++.h>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

const float eps = 1e-6;

//SVD function using eigen library
void SVD(MatrixXf &M, MatrixXf &U, MatrixXf &sing_vals, MatrixXf &V){
	JacobiSVD<MatrixXf> svd(M, ComputeThinU | ComputeThinV);
	U = svd.matrixU();
	V = svd.matrixV();
	sing_vals = svd.singularValues();
}

//Auxiliar matrix: M(i,j) has always the mean of values in column j of the initial X for all i
MatrixXf get_M0(MatrixXf &X, MatrixXf &W){
	MatrixXf M(X.rows(), X.cols());

	#pragma omp parallel for
	for(int j=0;j<X.cols();j++){
		float mean = 0.0;
		int quant = 0;

		#pragma omp parallel for reduction(+:mean)
		for(int i=0;i<X.rows();i++){
			if(W(i,j) == 1.0){
				mean += X(i,j);
				quant++;
			}
		} 

		mean /= quant;

		//#pragma omp parallel for
		for(int i=0;i<M.rows();i++) M(i,j) = mean;
	}
	
	return M;
}

void get_X0(MatrixXf &X, MatrixXf &W, MatrixXf &M0){
	//Threading to the two nested loops
	#pragma omp parallel for collapse(2)
	for(int i=0;i<X.rows();i++){
		for(int j=0;j<X.cols();j++){
			if(W(i,j) == 0.0) X(i,j) = M0(i,j);
		}
	}
}

MatrixXf get_fitted_matrix(MatrixXf &X, MatrixXf &U, MatrixXf &sing_vals, MatrixXf &V, int ncp){
	MatrixXf mi = MatrixXf::Zero(X.rows(), X.cols());

	//Threading to the three nested loops
	#pragma omp parallel for collapse(3)
	for(int i=0;i<X.rows();i++){
		for(int j=0;j<X.cols();j++){
			for(int s=0;s<ncp;s++){
				mi(i,j) += sing_vals(s,0)*U(i,s)*V(j,s);
			}
		}
	}

	return mi;
}

void get_X(MatrixXf &X, MatrixXf &W, MatrixXf &M, MatrixXf &mi){
	MatrixXf ones(X.rows(), X.cols());

	//Threading to the two nested loops
	#pragma omp parallel for collapse(2)
	for(int i=0;i<X.rows();i++){
		for(int j=0;j<X.cols();j++){
			ones(i,j) = 1.0;
		}
	}

	X = (W.array())*((X - M).array()) + ((ones - W).array())*(mi.array());
	X += M;
}

//Auxiliar matrix: M(i,j) has always the mean of values in column j of the current X for all i
MatrixXf get_M(MatrixXf &X, MatrixXf &W){
	MatrixXf M(X.rows(), X.cols());

	#pragma omp parallel for
	for(int j=0;j<X.cols();j++){
		float mean = 0.0;

		#pragma omp parallel for reduction(+:mean)
		for(int i=0;i<X.rows();i++) mean += X(i,j);

		mean /= X.rows();

		#pragma omp parallel for
		for(int i=0;i<M.rows();i++) M(i,j) = mean;
	}
	
	return M;
}

bool converged(MatrixXf &prev_mi, MatrixXf &mi){
	float sum = 0.0;

	//Threading to the two nested loops
	#pragma omp parallel for reduction(+:sum) collapse(2)
	for(int i=0;i<mi.rows();i++){
		for(int j=0;j<mi.cols();j++){
			sum += (prev_mi(i,j) - mi(i,j))*(prev_mi(i,j) - mi(i,j));
		}
	}

	return sum <= eps;
}

void impute_pca(MatrixXf &X, MatrixXf &W, int ncp){
	if(ncp < 1 || ncp > min(X.rows(), X.cols())){
		cout << "Please, insert a valid value for the number of components used!" << endl;
		return;
	}

	MatrixXf M = get_M0(X, W);
	get_X0(X, W, M);
	MatrixXf mi_prev, mi, U, sing_vals, V, diff;
	int it_number = 0;

	while(true){
		it_number++;

		if(it_number == 1){
			diff = X - M;
			SVD(diff, U, sing_vals, V);
			mi_prev = get_fitted_matrix(M, U, sing_vals, V, ncp);
			get_X(X, W, M, mi_prev);
			get_M(X,W);
		}
		else{
			diff = X - M;
			SVD(diff, U, sing_vals, V);
			mi = get_fitted_matrix(M, U, sing_vals, V, ncp);

			if(converged(mi_prev, mi)) break;

			mi_prev = mi;
			get_X(X, W, M, mi);
			get_M(X,W);
		}
	}
}

int estim_ncp(MatrixXf &X, MatrixXf &W){
	float min_msep = -1.0;
	int min_ncp;

	//Different iterations in the loop may take different time to execute
	//So, I used dynamic schedule
	//nonmonotonic to profit the fact that each thread can execute chunks in an unspecified order
	#pragma omp parallel for schedule(nonmonotonic:dynamic)
	for(int ncp = 1; ncp <= min(X.rows(), X.cols()); ncp++){
		float cur_msep = 0.0;
		MatrixXf imputed_X(X.rows(), X.cols());

		//Threading to the two nested loops
		#pragma omp parallel for collapse(2)
		for(int i=0;i<X.rows();i++){
			for(int j=0;j<X.cols();j++){
				MatrixXf X_copy = X;
				MatrixXf W_copy = W;
				W_copy(i,j) = 0.0;
				impute_pca(X_copy, W_copy, ncp);
				imputed_X(i,j) = X_copy(i,j);
			}
		}

		//Threading to the two nested loops
		#pragma omp parallel for reduction(+:cur_msep) collapse(2)
		for(int i=0;i<X.rows();i++){
			for(int j=0;j<X.cols();j++){
				cur_msep += (X(i,j) - imputed_X(i,j))*(X(i,j) - imputed_X(i,j));
			}
		}

		if(min_msep == -1.0 || cur_msep < min_msep){
			min_msep = cur_msep;
			min_ncp = ncp;
		}
	}

	return min_ncp;
}

int main()
{
	///Example
	MatrixXf X(4,3);
	X(0,0) = 2.0; X(1,0) = 3.0; X(2,0) = 0.0; X(3,0) = 6.0;
	X(0,1) = 1.0; X(1,1) = 0.0; X(2,1) = 4.0; X(3,1) = 0.0;
	X(0,2) = 4.0; X(1,2) = 1.0; X(2,2) = 3.0; X(3,2) = 0.0;

	cout << "X before: " << endl << X << endl;

	MatrixXf W(4,3);
	W(0,0) = 1.0; W(1,0) = 1.0; W(2,0) = 0.0; W(3,0) = 1.0;
	W(0,1) = 1.0; W(1,1) = 0.0; W(2,1) = 1.0; W(3,1) = 0.0;
 	W(0,2) = 1.0; W(1,2) = 1.0; W(2,2) = 1.0; W(3,2) = 0.0;

	auto start = std::chrono::high_resolution_clock::now();
	int ncp = estim_ncp(X, W);
	cout << "Best number of components: " << estim_ncp(X, W) << endl;
	auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Time taken: " << elapsed.count() << " s" << endl;
    impute_pca(X, W, ncp);
    cout << "X after: " << endl << X << endl;
}