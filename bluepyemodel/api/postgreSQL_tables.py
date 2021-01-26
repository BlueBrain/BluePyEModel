""" SQL commands used to create the tables"""

def_extraction_targets = """CREATE TABLE {}_extraction_targets (
emodel VARCHAR (20) NOT NULL,
species VARCHAR (20) NOT NULL,
ecode VARCHAR (20) NOT NULL,
targets smallint[] NOT NULL,
tolerance smallint[] NOT NULL,
efeatures JSONB NOT NULL,
location VARCHAR (20) NOT NULL,
threshold bool NOT NULL
);"""

def_extraction_files = """CREATE TABLE {}_extraction_files (
emodel VARCHAR (20) NOT NULL,
species VARCHAR (20) NOT NULL,
cell_id VARCHAR (50) NOT NULL,
path text NOT NULL,
ecode VARCHAR (20) NOT NULL,
t_unit VARCHAR (10),
v_unit VARCHAR (10),
i_unit VARCHAR (10),
ton real,
toff real,
tmid real,
tmid2 real,
tend real,
liquid_junction_potential real,
PRIMARY KEY (path)
);"""


def_extraction_efeatures = """CREATE TABLE {}_extraction_efeatures (
emodel VARCHAR (20) NOT NULL,
species VARCHAR (20) NOT NULL,
name text NOT NULL,
protocol text,
mean real NOT NULL,
std real NOT NULL,
location VARCHAR (20)
);"""

def_extraction_protocols = """CREATE TABLE {}_extraction_protocols (
emodel VARCHAR (20) NOT NULL,
species VARCHAR (20) NOT NULL,
name text NOT NULL,
path text,
definition JSONB
);"""

def_optimisation_targets = """CREATE TABLE {}_optimisation_targets (
emodel VARCHAR (20) NOT NULL,
species VARCHAR (20) NOT NULL,
ecode VARCHAR (20) NOT NULL,
target smallint NOT NULL,
efeatures JSONB,
location VARCHAR (20) NOT NULL,
extra_recordings JSONB[],
type text NOT NULL
);"""

def_morphologies = """CREATE TABLE {}_morphologies (
name text NOT NULL,
path text NOT NULL,
emodel VARCHAR (20) NOT NULL,
species VARCHAR (20) NOT NULL,
seclist_names text[],
secarray_names text[],
sec_index int
);"""

def_optimisation_morphology = """CREATE TABLE {}_optimisation_morphology (
emodel VARCHAR (20) NOT NULL,
species VARCHAR (20) NOT NULL,
name text NOT NULL
);"""

def_optimisation_parameters = """CREATE TABLE {}_optimisation_parameters (
emodel VARCHAR (20) NOT NULL,
species VARCHAR (20) NOT NULL,
name text NOT NULL,
mechanism text,
value real[] NOT NULL,
locations text[],
distribution text
);"""

def_optimisation_distributions = """CREATE TABLE {}_optimisation_distributions (
name text NOT NULL,
function text NOT NULL,
parameters text[],
soma_ref_location real,
PRIMARY KEY (name)
);"""

def_mechanisms_path = """CREATE TABLE {}_mechanisms_path (
name text NOT NULL,
path text NOT NULL,
stochastic bool NOT NULL,
PRIMARY KEY (name)
);"""

def_models = """CREATE TABLE {}_models (
emodel VARCHAR (20) NOT NULL,
species VARCHAR (20) NOT NULL,
fitness real,
parameters JSONB,
scores JSONB,
scores_validation JSONB,
validated bool,
optimizer text,
seed smallint,
githash VARCHAR (20)
);"""

def_validation_targets = """CREATE TABLE {}_validation_targets (
emodel VARCHAR (20) NOT NULL,
species VARCHAR (20) NOT NULL,
ecode VARCHAR (20) NOT NULL,
target smallint NOT NULL,
efeatures JSONB,
location VARCHAR (20) NOT NULL,
extra_recordings JSONB[],
type text NOT NULL
);"""
