@ECHO OFF
set /p PATH_TO_CLOUDCOMPARE= Enter path to Cloud Compare: 
cd /D %PATH_TO_CLOUDCOMPARE%

set /p PATH_TO_POINTCLOUDS= Enter path to point clouds: 

for /d %%i in (%PATH_TO_POINTCLOUDS%\*) do ( ::Goes through each directory in given path
	echo %%i
	for %%f in (%%i\*) do ( ::Goes through each file in the directory
		set /p val=<%%f
  		echo "path: %%f"
		::CloudCompare -SILENT -O %%f -C_EXPORT_FMT ASC -EXT CSV -PREC 6 -SEP COMMA -SS OCTREE 8 ::SubSampling of Octree
		CloudCompare -SILENT -O %%f -C_EXPORT_FMT ASC -EXT CSV -PREC 6 -SEP COMMA -SS SPATIAL 0.09 ::SubSampling of Spatial
		del %%f
	)
)
echo =======================================================
echo 			Complete
echo =======================================================
PAUSE