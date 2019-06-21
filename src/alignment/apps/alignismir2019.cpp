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
#include <iomanip>
using namespace std;

#include "libmusical.h"
#include "ISMIR2019Alignment.h"

int main(int argc, char * argv[]) {

	if ( argc != 3 ) {
		cerr << "Usage: alignnlb <file1.json> <file2.json>" << endl;
		exit(1);
	}

	clog << "Reading from" << argv[1] << " and " << argv[2] << endl;


	// Get a JSON string for sequence 1 from a file
	// Create a Reader object for the JSON string
	musical::ISMIR2019JSONReader mr1(new musical::JSONFileSource(argv[1]));

	// Ask the Reader to generate the Sequence
	musical::ISMIR2019Sequence * seq1 =
		static_cast<musical::ISMIR2019Sequence*>(mr1.generateSequence());

	// Do the same for sequence 2
	musical::ISMIR2019JSONReader mr2(new musical::JSONFileSource(argv[2]));
	musical::ISMIR2019Sequence * seq2 =
		static_cast<musical::ISMIR2019Sequence*>(mr2.generateSequence());

	seq1->dump_stdout();
	seq2->dump_stdout();

	// Encapsulate the two sequences in a Sequences object
	musical::ISMIR2019Sequences * seqs = new musical::ISMIR2019Sequences(seq1,seq2);
	//seqs->setPitch40Shift(seqs->getNthComputedPitch40Shift(0));

	// Create a similarity rater
	musical::ISMIR2019OptiSimilarityRater * sr = new musical::ISMIR2019OptiSimilarityRater();
	//musical::ISMIR2019PitchbandIMASimilarityRater * sr = new musical::ISMIR2019PitchbandIMASimilarityRater();
	//musical::ISMIR2019AlwaysOneSimilarityRater * sr = new musical::ISMIR2019AlwaysOneSimilarityRater();

	// Create a gap rater
	musical::ConstantAffineGapRater * gr = new musical::ConstantAffineGapRater((double)80*-0.01,(double)20*-0.01);

	// Create an alignment algorithm
	//musical::AffineGlobalAligner nw = musical::AffineGlobalAligner(seqs, sr, gr);
	musical::LinearLocalAligner nw = musical::LinearLocalAligner(seqs, sr, gr);

	// Debug
	nw.setFeedback(true);

	// Do the alignment
	nw.doAlign();

	// Print the alignment to stdout
	musical::AlignmentVisualizer av(seqs);
	av.basicStdoutReport();
	av.toGnuPlot("alignment-"+seq1->getName()+"-"+seq2->getName());

	double normalizedscore = seqs->getScore() / min(seq1->size(),seq2->size());
	clog << "         Aligner: " << nw.getName() << endl;
	clog << "       Gap Rater: " << gr->getName() << endl;
	clog << "   Pitch40 Shift: " << seqs->getPitch40Shift() << endl;
	clog << "   Pitch12 Shift: " << seqs->getPitch12Shift() << endl;
	clog << "           Score: " << seqs->getScore() << endl;
	clog << "Normalized score: " << normalizedscore << endl;
	clog << "        Distance: " << 1.0 - normalizedscore << endl;

	// free memory
	delete seq1;
	delete seq2;
	delete seqs;
	delete gr;
	delete sr;

	return 0;
}
