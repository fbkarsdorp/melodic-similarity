/*
Author: Peter van Kranenburg (peter.van.kranenburg@meertens.knaw.nl)
Copyright 2011 Meertens Institute (KNAW)

This file is part of libmusical.

libmusical is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

libmusical is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with libmusical.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
using namespace std;

#include <sys/time.h>
#include <time.h>

#include "libmusical.h"
#include "ISMIR2019Alignment.h"

/**
 * returns the current time in microsecs.
 * Doesn't work in linux for some reason.
 */
inline float datemicro() {
     struct timeval tv;
     struct timezone tz;
     struct tm *tm;
     gettimeofday(&tv, &tz);
     tm=localtime(&tv.tv_sec);
     return tm->tm_sec + 60*tm->tm_min + 60*60*tm->tm_hour + (float)tv.tv_usec/1000000.0;
}

int main(int argc, char * argv[]) {

	if (argc != 5 ) {
		cerr << "Usage: ismir2019distmat <flist1> <flist2> <configuration> <outputfile>"
	}

	bool distmat = true; //create distmat file?
	double * thedistmat;

	ifstream listfile1, listfile2;

	ofstream outfile;

	ifstream seqfile1, seqfile2;
	string seq1name,seq2name;
	string seq1,seq2;

	listfile1.open(argv[1]);
	listfile2.open(argv[2]);
	vector<musical::ISMIR2019Sequence *> seqs1;
	vector<musical::ISMIR2019Sequence *> seqs2;
	cout << "Reading sequences 1";
	while (getline(listfile1, seq1name)) {
		musical::ISMIR2019JSONReader mr1(new musical::JSONFileSource(seq1name));
		seqs1.push_back(static_cast<musical::ISMIR2019Sequence*>(mr1.generateSequence()));
		//if ( seqs1.size() % 1000 == 0 ) cout << seq1name << endl;
		if ( seqs1.size() % 1000 == 0 ) cout << "." << flush;
	}
	cout << endl;

	cout << "Reading seqeunces 2";
	while (getline(listfile2, seq2name)) {
		musical::ISMIR2019JSONReader mr2(new musical::JSONFileSource(seq2name));
		seqs2.push_back(static_cast<musical::ISMIR2019Sequence*>(mr2.generateSequence()));
		//if ( seqs2.size()%1000 == 0 ) cout << seq2name << endl;
		if ( seqs2.size() % 1000 == 0 ) cout << "." << flush;
	}
	cout << endl;
	listfile1.close();
	listfile2.close();

	//dump sequence;
	//cout << "sequence: " << endl;
	//seq->dump_stdout();

	string distmatfile = argv[3];

	if (distmat) {
		outfile.open(distmatfile.c_str());
		outfile << "recnr";

		for ( unsigned int i = 0; i < seqs2.size(); i++ ) {
			outfile << "\t" << seqs2[i]->getName();
		}
		outfile << endl;

		//prepare memory for the distmat
		thedistmat = (double *)malloc(seqs1.size()*seqs2.size()*sizeof(double));

	}

	int size1 = seqs1.size();
	int size2 = seqs2.size();

	float begin = datemicro();

	for(int i = 0; i<size1; i++) {
		cout << i << ": " << seqs1[i]->getName() << endl;
		//#pragma omp parallel for
		for(int j=0; j<size2; j++) {
			if ( j%1000 == 0 ) cout << "." << flush;
			if ( seqs1[i]->size() == 0 || seqs2[j]->size() == 0 ) {
				if (distmat) thedistmat[i*size2+j] = 100.0;
			} else {
				musical::ISMIR2019Sequences * seqs = new musical::ISMIR2019Sequences(seqs1[i],seqs2[j]);
				//musical::ISMIR2019OptiSimilarityRater * sr = new musical::ISMIR2019OptiSimilarityRater();
				musical::ISMIR2019ExactPitch40SimilarityRater * sr = new musical::ISMIR2019ExactPitch40SimilarityRater();
				musical::ConstantAffineGapRater * gr = new musical::ConstantAffineGapRater(-0.6, -0.2);
				musical::AffineGlobalAligner nw = musical::AffineGlobalAligner(seqs, sr , gr);
				nw.doAlign();
				double normalizedscore = seqs->getScore() / min(seqs1[i]->size(),seqs2[j]->size());
				if (distmat) thedistmat[i*size2+j] = 1.0 - normalizedscore;
				delete seqs;
				delete gr;
				delete sr;
			}
		}
		cout << endl;
	}

	float end = datemicro();

	cout << "   total time : " << end - begin << endl;
	cout << "time per query: " << (end - begin)/(float)seqs1.size() << endl;

	if (distmat) {
		cout << "Writing " << distmatfile << endl;
		for ( int i = 0; i < size1; i++ ) {
			outfile << seqs1[i]->getName();
			for ( int j = 0; j < size2; j++ ) {
				double dist = thedistmat[i*size2+j];
				outfile << "\t" << dist;
			}
			outfile << endl;
		}
		free(thedistmat);
	}

	//delete te sequences:
	for (int i=0; i<size1; i++) delete seqs1[i];
	for (int j=0; j<size2; j++) delete seqs2[j];

	return 0;
}
