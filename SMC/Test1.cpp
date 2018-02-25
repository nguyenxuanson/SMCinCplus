#include <iostream>
#include <algorithm>
#include <ctime>
#include <Eigen/Eigen>

#include <boost/chrono.hpp>
#include <boost/config/suffix.hpp>
#include <boost/format.hpp>

//http://www.boost.org/doc/libs/1_55_0b1/libs/iostreams/doc/index.html
//http://www.boost.org/doc/libs/1_56_0/doc/html/program_options/tutorial.html
#include <boost/program_options.hpp>

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/non_central_chi_squared.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include <boost/nondet_random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/shuffle_order.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_int_distribution.hpp>
#include <boost/random/variate_generator.hpp>

using namespace std;
using namespace Eigen;
namespace po = boost::program_options;

typedef std::vector<size_t> VectorSize_t;//aligned vector

template<typename T>
class maybe_t
{
	bool isNothing;
	T val;
public:
	maybe_t() :isNothing(true), val(T()){}
	maybe_t(T const& t) :isNothing(false), val(t){}
	bool nothing()const{ return isNothing; }
	T const& just()const{ return val; }
};

double cnorm(double x)
{
	// define a standard normal distribution:
	static const boost::math::normal norm;
	return cdf(norm, x);
}

class SdeParams
{
public:
	SdeParams
		(double S
		, double mu
		, double rd
		, double sigma
		)
		: m_S(S)
		, m_mu(mu)
		, m_rd(rd)
		, m_sigma(sigma)
		, m_rf(rd - mu)
	{};
	double spot()const{ return m_S; }
	double rd()const{ return m_rd; }
	double rf()const{ return m_rf; }
	double vol()const{ return m_sigma; }
	double drift()const{ return m_mu; }
private:
	double m_S;
	double m_mu;
	double m_rd;
	double m_sigma;
	double m_rf;
};

std::ostream& operator<<(std::ostream& os, SdeParams const& obj)
{
	os << "SdeParams\t"
		<< boost::str(boost::format("spot = %.16f\t") % obj.spot())
		<< boost::str(boost::format("rd = %.16f\t") % obj.rd())
		<< boost::str(boost::format("rf = %.16f\t") % obj.rf())
		<< boost::str(boost::format("drift = %.16f\t") % obj.drift())
		<< boost::str(boost::format("vol = %.16f\t") % obj.vol())
		<< "\n"
		;
	return os;
}

size_t effectiveSize(size_t const& nSteps, std::vector<size_t>const& vecSaveIndex)
{
	size_t const countMax = vecSaveIndex.size();
	if (countMax>0)
	{
		BOOST_ASSERT(vecSaveIndex.front()>0);
		for (size_t i = 1; i<countMax; ++i)
		{
			BOOST_ASSERT(vecSaveIndex[i]>vecSaveIndex[i - 1]);
		}
	}

	size_t count = 0;
	for (; count<countMax;)
	{
		if (vecSaveIndex[count] <= nSteps)
			++count;
		else
			break;
	}

	return count;
}

// http://www.boost.org/doc/libs/1_47_0/libs/random/example/random_demo.cpp
// http://www.boost.org/doc/libs/1_47_0/boost/random/uniform_int.hpp

template<typename RNG, typename IntType>
struct ReshuffleEng : std::unary_function<IntType, IntType>
{
	IntType operator()(IntType i)
	{
		return m_rng(*m_state, i);
	}
	ReshuffleEng(RNG * const state) : m_state(state) {}
private:
	RNG * const m_state;
	boost::uniform_int<> m_rng;
};

struct ProblemData
{
	ProblemData
	(double K
	, double T
	, bool isCall
	, maybe_t<double> barU
	, maybe_t<double> barD
	)
	: m_K(K)
	, m_T(T)
	, m_isCall(isCall)
	, m_barU(barU)
	, m_barD(barD)
	, m_CG()
	, m_CL()
	, m_RC()
	{};
	double m_K;
	double m_T;
	bool m_isCall;
	maybe_t<double> m_barU;
	maybe_t<double> m_barD;
	//////////////////////
	std::vector<double> m_Te; // fixing dates
	std::vector<double> m_Td; // delivery dates
	// vector of payoff?
	maybe_t<double> m_CG; // taget client gain
	maybe_t<double> m_CL; // taget client loss
	maybe_t<double> m_RC; // taget client range count
};

void initShuffling(VectorSize_t& shufflingInitial)
{
	for (size_t i = 0; i<shufflingInitial.size(); ++i)
		shufflingInitial[i] = i;
}

//! Path generation, keeping only two slice, not a cube
//!
//! Need to use boost if optimize for memory, Cortex does not store state of random generator and every Draw give the same number
class EngineMC
{
public:
	typedef boost::random::mt19937 RngSmc_t; // mt11213b

	//typedef boost::uniform_int<> NumberDistribution; //typedef boost::uniform_real<> NumberDistribution;
	typedef boost::random::normal_distribution<double> NormalDistribution;
	typedef boost::random::mt19937 RandomNumberGenerator;
	typedef boost::variate_generator<RandomNumberGenerator&, NormalDistribution> NormalGenerator;
private:
	SdeParams const m_SdeParams;
	size_t const m_nPaths;
	std::vector<double>const m_timeSteps;
	size_t const m_nSteps;
	maybe_t<double>const m_levelESS;
	size_t const m_seed;
	NormalDistribution m_NormalDistribution;
	RandomNumberGenerator m_RandomNumberGenerator;
	NormalGenerator m_RandGen;
	//////////////////////////////////////////////
	VectorXd m_stateSpot;
	VectorXd m_stateBars;
	VectorXd m_vecNoise;
	//////////////////////////////////////////////
	double const mu;
	double const vol;
	double const halfVol2;
	//////////////////////////////////////////////
	boost::random_device m_rd;
	RngSmc_t m_RngSmc;
	ReshuffleEng<RngSmc_t, size_t> m_RndSmcFun;
	VectorSize_t m_Shuffling;
	size_t m_ShufflingMarker;// number of alive state
private:
	double computeESS(VectorXd const& stateBars)
	{
		m_ShufflingMarker = 0;
		for (size_t i = 0; i<stateBars.size(); ++i)
		{
			if (stateBars[i]>0)
			{
				m_Shuffling[m_ShufflingMarker] = i;
				++m_ShufflingMarker;
			}
		}
		return static_cast<double>(m_ShufflingMarker) / stateBars.size();
	}

	void resampling(VectorXd& stateBars, VectorXd& stateSpot)
	{
		std::random_shuffle(m_Shuffling.begin(), m_Shuffling.begin() + m_ShufflingMarker, m_RndSmcFun);

		size_t const modulo = stateSpot.size() / m_ShufflingMarker;
		size_t const remainder = stateSpot.size() % m_ShufflingMarker;
		for (size_t i = 1; i < modulo; ++i)
			std::copy
			(m_Shuffling.begin()
			, m_Shuffling.begin() + m_ShufflingMarker
			, m_Shuffling.begin() + m_ShufflingMarker*i
			);
		std::copy
			(m_Shuffling.begin()
			, m_Shuffling.begin() + remainder
			, m_Shuffling.begin() + m_ShufflingMarker*modulo
			);

		for (size_t i = 0; i < stateSpot.size(); ++i)
		{
			stateBars[i] = stateSpot[m_Shuffling[i]];
		}
		stateSpot = stateBars;
		for (size_t i = 0; i < stateBars.size(); ++i)
		{
			stateBars[i] = 1;
		}
	}
public:
	EngineMC
		(SdeParams const& sdeParams
		, size_t const nPaths
		, std::vector<double>const& timeSteps
		, maybe_t<double>const& levelESS // effective sampling size threshold
		, size_t const seed = 0// not used
		)
		: m_SdeParams(sdeParams)
		, m_nPaths(nPaths)
		, m_timeSteps(timeSteps)
		, m_nSteps(timeSteps.size())
		, m_levelESS(levelESS)
		, m_seed(seed)
		, m_NormalDistribution()
		, m_RandomNumberGenerator()
		, m_RandGen(m_RandomNumberGenerator, m_NormalDistribution)
		///////////////////////////////////////////
		, m_stateSpot(VectorXd::Constant(nPaths, 0.0))
		, m_stateBars(VectorXd::Constant(nPaths, 0.0))
		, m_vecNoise (VectorXd::Constant(nPaths, 0.0))
		///////////////////////////////////////////
		, mu(m_SdeParams.drift())
		, vol(m_SdeParams.vol())
		, halfVol2(vol*vol / 2)
		///////////////////////////////////////////
		, m_rd()//initialize randome decive
		//, m_RngSmc(m_rd())// random seed (from machine state)
		, m_RngSmc(0)// alternatively pass a seed
		, m_RndSmcFun(&m_RngSmc)
		, m_Shuffling(nPaths)
		, m_ShufflingMarker(0)
	{
		// https://software.intel.com/sites/products/documentation/hpc/mkl/vslnotes/8_4_8_MT2203.htm
		// it has 6024  independent random number sequences.
		m_RandomNumberGenerator.seed(m_seed);
	};
public:
	VectorXd const& noises()const
	{
		return m_vecNoise;
	}
	MatrixXd const& move_forward
		(ProblemData const& problemData
		, std::vector<size_t>const& vecSaveIndex
		, MatrixXd& vecStateSpot
		, MatrixXd& vecStateProb
		, MatrixXd& vecStateCGA
		, MatrixXd& vecStateCGE
		, MatrixXd& vecStateCLA
		, MatrixXd& vecStateCLE
		, MatrixXd& vecStateRCA
		, MatrixXd& vecStateRCE
		, double& normalizedProba
		)
	{
		double const lnBarU = problemData.m_barU.nothing() ? 0 : std::log(problemData.m_barU.just());
		double const lnBarD = problemData.m_barD.nothing() ? 0 : std::log(problemData.m_barD.just());
		// allocate storage for states
		size_t const nbStateSaved = effectiveSize(m_nSteps, vecSaveIndex);
		vecStateSpot = MatrixXd::Constant(m_nPaths, nbStateSaved, 0);
		vecStateProb = MatrixXd::Constant(m_nPaths, nbStateSaved, 1);
		size_t count = 0;

		//initialize MC state at pricing date
		normalizedProba = 1;
		VectorXd& stateSpot = m_stateSpot;
		stateSpot.setConstant(std::log(m_SdeParams.spot()));
		VectorXd& stateBars = m_stateBars;
		stateBars.setConstant(1);
		
		for (size_t k = 0; k < m_nSteps; ++k)
		{
			for (size_t i = 0; i < m_nPaths; ++i)
				m_vecNoise[i] = m_RandGen();

			VectorXd const& noiseNext = m_vecNoise;
			double const& dt = m_timeSteps[k];
			double const loc = (mu - halfVol2)*dt;
			double const scale = vol*std::sqrt(dt);

			stateSpot += VectorXd::Constant(stateSpot.size(), loc) + noiseNext*scale;

			if (!problemData.m_barU.nothing())
			{
				stateBars = stateBars.array() * (stateSpot.array() < lnBarU).cast<double>();
			}
			if (!problemData.m_barD.nothing())
			{
				stateBars = stateBars.array() * (stateSpot.array() > lnBarD).cast<double>();
			}

			if (!m_levelESS.nothing())
			{
				double const ess = computeESS(stateBars);
				BOOST_ASSERT_MSG(ess > 0.1, "Survival Prob is too small");
				if (ess < m_levelESS.just())
				{
					normalizedProba *= ess;
					resampling(stateBars, stateSpot);
				}
			}

			if (k + 1 == vecSaveIndex[count])
			{
				vecStateSpot.col(count) = stateSpot;
				vecStateProb.col(count) = stateBars;
				++count;
			}
		}

		return vecStateSpot;
	};
};

double evalExpectation
	( ProblemData const& problemData
	, SdeParams const& sdeParams
	, VectorXd& vecPrice
	, VectorXd& vecSpot
	, VectorXd& vecProb
	)
{
	double const& T = problemData.m_T;
	double const& K = problemData.m_K;
	bool const& isCall = problemData.m_isCall;

	BOOST_ASSERT(vecSpot.size() == vecPrice.size());

	if (isCall)
	{
		vecPrice = vecProb.array()*(vecSpot.array().exp() - VectorXd::Constant(vecSpot.size(), K).array());
	}
	else
	{
		vecPrice = vecProb.array()*(VectorXd::Constant(vecSpot.size(), K).array() - vecSpot.array().exp());
	}

	double price = (vecPrice.cwiseMax(0)).sum();

	return price / vecSpot.size() / std::exp(sdeParams.rd()*T);
}

double price_Vanilla
	( ProblemData const& problemData
	, SdeParams const& sdeParams
)
{
	double const& T = problemData.m_T;
	double const& K = problemData.m_K;
	bool const& isCall = problemData.m_isCall;
	BOOST_ASSERT(T>0);
	double const totalVol = sdeParams.vol()*std::sqrt(T);
	double const tmp = (std::log(sdeParams.spot() / K) + sdeParams.drift()*T) / totalVol;
	double const d1 = tmp + totalVol / 2;
	double const d2 = tmp - totalVol / 2;
	double const fwd = sdeParams.spot()*std::exp(sdeParams.drift()*T);

	double priceVanillaTmp = 0;
	if (isCall)
	{
		priceVanillaTmp = std::exp(-sdeParams.rd()*T)*(fwd*cnorm(d1) - K*cnorm(d2));
	}
	else
	{
		priceVanillaTmp = std::exp(-sdeParams.rd()*T)*(K*cnorm(-d2) - fwd*cnorm(-d1));
	}
	return priceVanillaTmp;
}

void worker
(ProblemData const& problemData
, SdeParams const& sdeParams
, size_t const& nPaths
, std::vector<double>const& timeSteps
, size_t const& seed
, size_t const& nSequences
, maybe_t<double>const& levelESS
)
{
	std::cout << "START\n";
	boost::chrono::steady_clock::time_point start = boost::chrono::steady_clock::now();

	std::vector<size_t> vecSaveIndex(1, timeSteps.size());
	MatrixXd vecStateSpot;
	MatrixXd vecStateProb;
	// these two states are used to construct new probability for target redemption product
	MatrixXd vecStateCGA; // accumulated client gain value of each path: exact value per path given from realization and option definition
	MatrixXd vecStateCGE; // expected client gain value of each path: our guess on future value given from realization and option definition

	MatrixXd vecStateCLA; // accumulated client loss value of each path: exact value per path given from realization and option definition
	MatrixXd vecStateCLE; // expected client loss value of each path: our guess on future value given from realization and option definition
	// prolong range count from integer to real
	MatrixXd vecStateRCA; // accumulated range count value of each path: exact value per path given from realization and option definition
	MatrixXd vecStateRCE; // expected range count value of each path: our guess on future value given from realization and option definition

	VectorXd vecPriceNum=VectorXd::Constant(nSequences, 0.0);
	for (size_t sqno = 0; sqno<nSequences; ++sqno)
	{
		EngineMC engineMC
			(sdeParams
			, nPaths
			, timeSteps
			, levelESS
			, sqno
			);
		double normalizedProba = 1;
		engineMC.move_forward
			(problemData
			, vecSaveIndex
			, vecStateSpot
			, vecStateProb
			, vecStateCGA
			, vecStateCGE
			, vecStateCLA
			, vecStateCLE
			, vecStateRCA
			, vecStateRCE
			, normalizedProba
			);

		VectorXd vecPrice = VectorXd::Constant(nPaths, 0.0);
		VectorXd vecSpot = vecStateSpot.col(vecStateSpot.cols() - 1);
		VectorXd vecProb = vecStateProb.col(vecStateProb.cols() - 1);
		
		vecPriceNum[sqno] = normalizedProba * evalExpectation(problemData, sdeParams, vecPrice, vecSpot, vecProb);
	}

	//double const priceAna = price_Vanilla(problemData, sdeParams);

	double priceNum = vecPriceNum.sum();
	double priceNum2 = vecPriceNum.squaredNorm();

	priceNum /= nSequences;
	double const priceNum_var = (priceNum2 - nSequences*priceNum*priceNum) / (nSequences - 1);
	double const priceNum_std = std::sqrt(priceNum_var);

	boost::chrono::duration<double> sec = boost::chrono::steady_clock::now() - start;

	std::cout
		//<< "\t" << boost::str(boost::format("priceAna =%.16f\t") % priceAna)
		<< "\t" << boost::str(boost::format("priceNum =%.16f\t") % priceNum)
		<< "\t" << boost::str(boost::format("priceNum_var =%.16f\t") % priceNum_var)
		<< "\t" << boost::str(boost::format("priceNum_std =%.16f\n") % priceNum_std)
		<< "\t" << boost::str(boost::format("EngineMC takes %.16f seconds\n") % sec.count())
		<< "END\n\n";
	;
}

int main0(int ac, char* av[])
{
	double S;
	double T;
	double K = 103;
	bool isCall;

	double barU;
	double barD;

	double mu;
	double rd;
	double sigma;

	size_t nPaths;
	size_t nSteps;
	size_t nSequences;
	size_t seed;

	maybe_t<double> levelESS;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help", "produce help message")
			("mu", po::value<double>(&mu)->default_value(0.02), "set mu")
			("rd", po::value<double>(&rd)->default_value(0.05), "set rd")
			("sigma", po::value<double>(&sigma)->default_value(0.1), "set sigma")
			/////////////////////
			("S", po::value<double>(&S)->default_value(100), "set S")
			("T", po::value<double>(&T)->default_value(1), "set T")
			("K", po::value<double>(&K)->default_value(103), "set K")
			/////////////////////
			("isCall", po::value<bool>(&isCall)->default_value(true), "set isCall")
			("barU", po::value<double>(&barU)->default_value(105), "set barU")
			("barD", po::value<double>(&barD)->default_value(97), "set barD")
			/////////////////////
			("nPaths", po::value<size_t>(&nPaths)->default_value(100000), "set nPaths")
			("nSteps", po::value<size_t>(&nSteps)->default_value(365), "set nSteps")
			("nSequences", po::value<size_t>(&nSequences)->default_value(100), "set nSequences")
			("seed", po::value<size_t>(&seed)->default_value(0), "set seed")
			("levelESS", po::value<double>()->default_value(0.7), "set levelESS")
			;

		po::variables_map vm;
		po::store(po::parse_command_line(ac, av, desc), vm);
		po::notify(vm);

		if (vm.count("help")) {
			cout << desc << "\n";
			return 1;
		}
		cout << "====================================\n";
		cout << "mu was set to " << vm["mu"].as<double>() << ".\n";
		cout << "rd was set to " << vm["rd"].as<double>() << ".\n";
		cout << "sigma was set to " << vm["sigma"].as<double>() << ".\n";

		cout << "S was set to " << vm["S"].as<double>() << ".\n";
		cout << "K was set to " << vm["K"].as<double>() << ".\n";
		cout << "T was set to " << vm["T"].as<double>() << ".\n";

		cout << "isCall was set to " << vm["isCall"].as<bool>() << ".\n";
		cout << "barU was set to " << vm["barU"].as<double>() << ".\n";
		cout << "barD was set to " << vm["barD"].as<double>() << ".\n";

		cout << "nPaths was set to " << vm["nPaths"].as<size_t>() << ".\n";
		cout << "nSteps was set to " << vm["nSteps"].as<size_t>() << ".\n";
		cout << "nSequences was set to " << vm["nSequences"].as<size_t>() << ".\n";
		cout << "seed was set to " << vm["seed"].as<size_t>() << ".\n";

		cout << "levelESS was set to " << vm["levelESS"].as<double>() << ".\n";
		if (vm.count("levelESS")) {
			levelESS = maybe_t<double>(vm["levelESS"].as<double>());
		}
		cout << "====================================\n";

	}
	catch (exception& e) {
		cerr << "error: " << e.what() << "\n";
		return 1;
	}
	catch (...) {
		cerr << "Exception of unknown type!\n";
	}

	SdeParams sdeParams01(S, mu, rd, sigma);

	std::vector<double> timeSteps(nSteps, T / nSteps);

	std::vector<ProblemData> problemDatas;
	problemDatas.push_back(ProblemData(K, T, isCall, maybe_t<double>(), maybe_t<double>()));// Vanilla
	problemDatas.push_back(ProblemData(K, T, isCall, maybe_t<double>(barU), maybe_t<double>(barD)));// DKO
	problemDatas.push_back(ProblemData(K, T, isCall, maybe_t<double>(), maybe_t<double>(barD)));// DO
	problemDatas.push_back(ProblemData(K, T, isCall, maybe_t<double>(barU), maybe_t<double>()));// UO

	for (size_t i = 0; i<problemDatas.size(); ++i)
	{
		std::cout << "NO   ESS\n"; worker(problemDatas[i], sdeParams01, nPaths, timeSteps, seed, nSequences, maybe_t<double>());
		std::cout << "WITH ESS\n"; worker(problemDatas[i], sdeParams01, nPaths, timeSteps, seed, nSequences, levelESS);
	}

	return 0;
}

int main1(int ac, char* av[])
{
	size_t const n = 10;
	VectorSize_t vec(n);
	initShuffling(vec);

	boost::random_device rd;

	EngineMC::RngSmc_t engine(0);
	//EngineMC::RngSmc_t engine(rd());
	//boost::random::mt19937 engine(std::time(NULL));

	//http://www.cplusplus.com/reference/algorithm/random_shuffle/

	ReshuffleEng<EngineMC::RngSmc_t, size_t> genFun(&engine);

	VectorSize_t vec1(vec);
	VectorSize_t vec2(vec);
	std::random_shuffle(vec1.begin(), vec1.end(), genFun);
	std::random_shuffle(vec2.begin(), vec2.end(), genFun);
	for (size_t i = 0; i<vec.size(); ++i)
	{
		std::cout << i << "\t" << vec1[i] << "\t" << vec2[i] << "\n";
	}

	return 0;
}

int main2(int ac, char* av[])
{
	int n =7000;
	MatrixXd A = MatrixXd::Random(n, n);
	MatrixXd B = MatrixXd::Random(n, 2);
	boost::chrono::steady_clock::time_point start = boost::chrono::steady_clock::now();
	MatrixXd X = A.lu().solve(B);
	boost::chrono::duration<double> sec = boost::chrono::steady_clock::now() - start;
	cout << "Relative error: " << (A*X - B).norm() / B.norm() << endl;
	cout << "Elapsed Time is: " << sec.count() << "s" << endl;

	return 0;
}

int main3(int ac, char* av[])
{
	Array3d v(1, 2, 3);
	cout << v << endl;
	cout << v.exp() << endl;

	cout << "\n";
	Vector3d u(2, - 3, 4), w(4, 2, 3);
	cout << u.cwiseMax(w) << endl;

	cout << "\n";
	cout << u.cwiseMax(0) << endl;

	cout << "\n";
	cout << u.array().exp() << endl;

	cout << "Flag againt 0\n";
	Vector3d f = (u.array() > 0).cast<double>();
	cout << u.array() * f.array();
	return 0;
}

int main(int ac, char* av[])
{
	return main2(ac, av);
}
