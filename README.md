# ML_FoCal_FinalProject
Final project for ML course using FoCal Data


To get datafiles:
  scp -T cernlogin@lxplus.cern.ch:/eos/experiment/alice/focal/TB_2026_Wk17_H2/data/disk2/data/Run557.ch2g .


To rootify data:
If first time rootifying, run 
  chmod +x run_rootifier.sh

Then run 
  ./run_rootifier.sh



Before syncing with the git, run 
  rm Rootified/intermediate/*.root
to empty the intermediate rootified results

And run 
  rm Data/*.ch2g
To remove the VERY large raw files
