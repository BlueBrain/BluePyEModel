python write_config_dict.py

# luigi --module bluepyemodel.tasks.emodel_creation.optimisation ExtractEFeatures --local-scheduler

# export IPYTHONDIR=${PWD}/.ipython
# export IPYTHON_PROFILE=opt
# echo $IPYTHON_PROFILE
# rm -rf ${IPYTHONDIR}
# ipcontroller --init --ip='*' --sqlitedb --profile=${IPYTHON_PROFILE} --ping=30000 &
# sleep 5
# srun ipengine --profile=${IPYTHON_PROFILE} --timeout=1000  &

luigi --module bluepyemodel.tasks.emodel_creation.optimisation Optimize --local-scheduler
