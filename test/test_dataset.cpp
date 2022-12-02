// This file is part of HSO: Hybrid Sparse Monocular Visual Odometry 
// With Online Photometric Calibration
//
// Copyright(c) 2021, Dongting Luo, Dalian University of Technology, Dalian
// Copyright(c) 2021, Robotics Group, Dalian University of Technology
//
// This program is highly based on the previous implementation 
// of SVO: https://github.com/uzh-rpg/rpg_svo
// and PL-SVO: https://github.com/rubengooj/pl-svo
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.


#include <string>

#include <iostream>

#include "hso/system.h"

int main(int argc, const char **argv)
{
    if (argc < 2)
    {
	std::cerr << std::endl << "Minimal Usage: ./test_dataset  image=PATH_TO_IMAGE_FOLDER  calib=PATH_TO_CALIBRATION"
		  << std::endl;
	return 1;
    }

    hso::System HSO(argc, argv);


    HSO.runFromFolder();

    printf("HSO_BenchmarkNode finished.\n");
    return 0;
}

