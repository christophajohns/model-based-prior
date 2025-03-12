import sqlite3
import json
from typing import List, Tuple, Dict, Optional, Literal
from dataclasses import dataclass
import datetime
import os
from botorch.test_functions.synthetic import SyntheticTestFunction
from botorch.utils.prior import UserPriorLocation
import torch
import tempfile

def adapt_date_iso(val):
    """Adapt datetime.date to ISO 8601 date."""
    return val.isoformat()

def adapt_datetime_iso(val):
    """Adapt datetime.datetime to timezone-naive ISO 8601 date."""
    return val.isoformat()

sqlite3.register_adapter(datetime.date, adapt_date_iso)
sqlite3.register_adapter(datetime.datetime, adapt_datetime_iso)

def convert_date(val):
    """Convert ISO 8601 date to datetime.date object."""
    return datetime.date.fromisoformat(val.decode())

def convert_datetime(val):
    """Convert ISO 8601 datetime to datetime.datetime object."""
    return datetime.datetime.fromisoformat(val.decode())

def convert_timestamp(val):
    """Convert ISO 8601 timestamp to datetime.datetime object."""
    return datetime.datetime.fromisoformat(val.decode())

sqlite3.register_converter("date", convert_date)
sqlite3.register_converter("datetime", convert_datetime)
sqlite3.register_converter("timestamp", convert_timestamp)

@dataclass
class OptimizationType:
    description: str
    id: Optional[int] = None
    created_at: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None

@dataclass
class Objective:
    dimensions: int
    bounds: List[Tuple[float, float]]
    description: str
    optimizers: List[Tuple[float, ...]]
    optimal_value: float
    id: Optional[int] = None
    created_at: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None

@dataclass
class PriorType:
    description: str
    id: Optional[int] = None
    created_at: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None

    NONE = "None"
    UNBIASED = "Unbiased"
    BIASED = "Biased"
    BIASED_CERTAIN = "BiasedCertain"
    UNBIASED_UNCERTAIN = "UnbiasedUncertain"
    UNBIASED_MORECERTAIN = "UnbiasedMoreCertain"
    UNBIASED_MOREUNCERTAIN = "UnbiasedMoreUncertain"
    UNBIASED_CERTAIN = "UnbiasedCertain"
    BIASED_MORECERTAIN = "BiasedMoreCertain"
    BIASED_MOREUNCERTAIN = "BiasedMoreUncertain"
    BIASED_UNCERTAIN = "BiasedUncertain"

@dataclass
class Prior:
    parameters: Dict
    prior_type_id: int
    id: Optional[int] = None
    created_at: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None

@dataclass
class PriorInjectionMethod:
    description: str  # e.g., 'ColaBO', 'piBO', 'None'
    id: Optional[int] = None
    created_at: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None

    COLABO = "ColaBO"
    PIBO = "piBO"
    NONE = "None"

@dataclass
class ExperimentConfig:
    optimization_type_id: int
    objective_id: int
    prior_id: int
    prior_injection_method_id: int
    seed: int
    num_trials: int
    num_initial_samples: int
    num_paths: int
    id: Optional[int] = None
    created_at: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None

@dataclass
class BOResult:
    experiment_id: int
    iteration: int
    parameters: List[Tuple[float, ...]]
    rating: float
    best_parameters: List[Tuple[float, ...]]
    best_rating: float
    id: Optional[int] = None
    created_at: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None

@dataclass
class PBOExperimentResult:
    experiment_id: int
    iteration: int
    parameters_preferred_id: int
    parameters_not_preferred_id: int
    best_rating: float
    parameters_best: Tuple[float, ...]
    id: Optional[int] = None
    created_at: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None

@dataclass
class PBOExperimentConfig:
    experiment_id: int
    parameters_id: int
    parameters: Tuple[float, ...]
    rating: float
    id: Optional[int] = None
    created_at: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None

class Database:
    _expected_tables = [
        ('optimization_types',),
        ('objectives',),
        ('prior_injection_methods',),
        ('prior_types',),
        ('priors',),
        ('experiment_configs',),
        ('bo_experiment_results',),
        ('pbo_experiment_results',),
        ('pbo_experiment_configs',)
    ]

    def __init__(self, path: str):
        """Initialize Database connection.
        
        Args:
            path (str): Path to SQLite database file
        """
        self.path = path
        db_exists = self._check_database(path)
        if not db_exists:
            self._create_database()

    def _check_database(self, path: str) -> bool:
        """Check whether a database with the correct schema exists."""
        if not os.path.exists(path):
            return False
        
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            if len(tables) < 8:
                return False
            
            if not all(table in tables for table in self._expected_tables):
                return False

        return True

    def _create_database(self):
        """Create database schema if it doesn't exist."""
        with self._connect() as conn:
            cursor = conn.cursor()
            
            # Create tables
            cursor.executescript('''
                CREATE TABLE IF NOT EXISTS optimization_types (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                                 
                CREATE TRIGGER IF NOT EXISTS update_optimization_types_timestamp
                AFTER UPDATE ON optimization_types
                BEGIN
                    UPDATE optimization_types SET updated_at=CURRENT_TIMESTAMP WHERE id=OLD.id;
                END;
                                 
                CREATE TABLE IF NOT EXISTS prior_injection_methods (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TRIGGER IF NOT EXISTS update_prior_injection_methods_timestamp
                AFTER UPDATE ON prior_injection_methods
                BEGIN
                    UPDATE prior_injection_methods SET updated_at=CURRENT_TIMESTAMP WHERE id=OLD.id;
                END;

                CREATE TABLE IF NOT EXISTS objectives (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dimensions INTEGER,
                    bounds TEXT,
                    description TEXT,
                    optimizers TEXT,
                    optimal_value REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                                 
                CREATE TRIGGER IF NOT EXISTS update_objectives_timestamp
                AFTER UPDATE ON objectives
                BEGIN
                    UPDATE objectives SET updated_at=CURRENT_TIMESTAMP WHERE id=OLD.id;
                END;

                CREATE TABLE IF NOT EXISTS prior_types (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                                 
                CREATE TRIGGER IF NOT EXISTS update_prior_types_timestamp
                AFTER UPDATE ON prior_types
                BEGIN
                    UPDATE prior_types SET updated_at=CURRENT_TIMESTAMP WHERE id=OLD.id;
                END;

                CREATE TABLE IF NOT EXISTS priors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parameters_json TEXT,
                    prior_type_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(prior_type_id) REFERENCES prior_types(id)
                );
                                 
                CREATE TRIGGER IF NOT EXISTS update_priors_timestamp
                AFTER UPDATE ON priors
                BEGIN
                    UPDATE priors SET updated_at=CURRENT_TIMESTAMP WHERE id=OLD.id;
                END;

                CREATE TABLE IF NOT EXISTS experiment_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    optimization_type_id INTEGER,
                    objective_id INTEGER,
                    prior_id INTEGER,
                    prior_injection_method_id INTEGER,
                    seed INTEGER,
                    num_trials INTEGER,
                    num_initial_samples INTEGER,
                    num_paths INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(optimization_type_id) REFERENCES optimization_types(id),
                    FOREIGN KEY(objective_id) REFERENCES objectives(id),
                    FOREIGN KEY(prior_id) REFERENCES priors(id),
                    FOREIGN KEY(prior_injection_method_id) REFERENCES prior_injection_methods(id)
                );
                                 
                CREATE TRIGGER IF NOT EXISTS update_experiment_configs_timestamp
                AFTER UPDATE ON experiment_configs
                BEGIN
                    UPDATE experiment_configs SET updated_at=CURRENT_TIMESTAMP WHERE id=OLD.id;
                END;

                CREATE TABLE IF NOT EXISTS bo_experiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    iteration INTEGER,
                    parameters TEXT,
                    rating REAL,
                    best_parameters TEXT,
                    best_rating REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(experiment_id) REFERENCES experiment_configs(id)
                );
                                 
                CREATE TRIGGER IF NOT EXISTS update_bo_experiment_results_timestamp
                AFTER UPDATE ON bo_experiment_results
                BEGIN
                    UPDATE bo_experiment_results SET updated_at=CURRENT_TIMESTAMP WHERE id=OLD.id;
                END;
                                 
                CREATE TABLE IF NOT EXISTS pbo_experiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    iteration INTEGER,
                    parameters_preferred_id INTEGER,
                    parameters_not_preferred_id INTEGER,
                    rating_best REAL,
                    parameters_best TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(experiment_id) REFERENCES experiment_configs(id),
                    FOREIGN KEY(parameters_preferred_id) REFERENCES pbo_experiment_results(id),
                    FOREIGN KEY(parameters_not_preferred_id) REFERENCES pbo_experiment_results(id)
                );
                                 
                CREATE TRIGGER IF NOT EXISTS update_pbo_experiment_results_timestamp
                AFTER UPDATE ON pbo_experiment_results
                BEGIN
                    UPDATE pbo_experiment_results SET updated_at=CURRENT_TIMESTAMP WHERE id=OLD.id;
                END;
                                 
                CREATE TABLE IF NOT EXISTS pbo_experiment_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER,
                    parameters_id INTEGER,
                    parameters TEXT,  -- tuple of floats
                    rating REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(experiment_id) REFERENCES experiment_configs(id),
                    FOREIGN KEY(parameters_id) REFERENCES pbo_experiment_results(id)
                );
                                 
                CREATE TRIGGER IF NOT EXISTS update_pbo_experiment_configs_timestamp
                AFTER UPDATE ON pbo_experiment_configs
                BEGIN
                    UPDATE pbo_experiment_configs SET updated_at=CURRENT_TIMESTAMP WHERE id=OLD.id;
                END;
            ''')

            # Add BO and PBO as default optimization types
            for description in ["BO", "PBO", "PriorSampling"]:
                cursor.execute("INSERT INTO optimization_types (description) VALUES (?)", (description,))

            # Add Unbiased, Biased, BiasedCertain, UnbiasedUncertain as default prior types
            bias = ["Unbiased", "Biased"]
            certainty = ["Certain", "Uncertain", "MoreCertain", "MoreUncertain"]
            prior_type_strings = [f"{b}{c}" for b in bias for c in certainty]
            for description in ["None", *bias, *prior_type_strings]:
                cursor.execute("INSERT INTO prior_types (description) VALUES (?)", (description,))

            # Add ColaBO, piBO, None as default prior injection methods
            for description in ["ColaBO", "piBO", "None"]:
                cursor.execute("INSERT INTO prior_injection_methods (description) VALUES (?)", (description,))


    def _connect(self):
        """Create a database connection."""
        return sqlite3.connect(self.path, 
                              detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)

    def add_optimization_type(self, opt_type: OptimizationType) -> int:
        """Add a new optimization type to the database.
        
        Args:
            opt_type (OptimizationType): The optimization type to add
            
        Returns:
            int: ID of the inserted optimization type
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO optimization_types (description) VALUES (?)",
                (opt_type.description,)
            )
            return cursor.lastrowid
        
    def add_prior_injection_method(self, method: PriorInjectionMethod) -> int:
        """Add a new prior injection method to the database.
        
        Args:
            method (PriorInjectionMethod): The prior injection method to add
            
        Returns:
            int: ID of the inserted prior injection method
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO prior_injection_methods (description) VALUES (?)",
                (method.description,)
            )
            return cursor.lastrowid

    def add_objective(self, objective: Objective) -> int:
        """Add a new objective function configuration to the database.
        
        Args:
            objective (Objective): The objective to add
            
        Returns:
            int: ID of the inserted objective
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO objectives 
                (dimensions, bounds, description, optimizers, optimal_value)
                VALUES (?, ?, ?, ?, ?)
            """, (
                objective.dimensions,
                json.dumps(objective.bounds),
                objective.description,
                json.dumps(objective.optimizers),
                objective.optimal_value,
            ))
            return cursor.lastrowid

    def add_prior_type(self, prior_type: PriorType) -> int:
        """Add a new prior type to the database.
        
        Args:
            prior_type (PriorType): The prior type to add
            
        Returns:
            int: ID of the inserted prior type
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO prior_types (description) VALUES (?)",
                (prior_type.description,)
            )
            return cursor.lastrowid

    def add_prior(self, prior: Prior) -> int:
        """Add a new prior to the database.
        
        Args:
            prior (Prior): The prior to add
            
        Returns:
            int: ID of the inserted prior
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO priors (parameters_json, prior_type_id) VALUES (?, ?)",
                (json.dumps(prior.parameters), prior.prior_type_id)
            )
            return cursor.lastrowid

    def add_experiment_config(self, config: ExperimentConfig) -> int:
        """Add a new experiment configuration to the database.
        
        Args:
            config (ExperimentConfig): Configuration for the experiment
            
        Returns:
            int: ID of the inserted experiment config
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO experiment_configs 
                (optimization_type_id, objective_id, prior_id, prior_injection_method_id,
                 seed, num_trials, num_initial_samples, num_paths)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                config.optimization_type_id,
                config.objective_id,
                config.prior_id,
                config.prior_injection_method_id,
                config.seed,
                config.num_trials,
                config.num_initial_samples,
                config.num_paths
            ))
            return cursor.lastrowid

    def add_bo_result(self, result: BOResult) -> int:
        """Add a new Bayesian Optimization result to the database.
        
        Args:
            result (BOResult): Result from a BO iteration
            
        Returns:
            int: ID of the inserted result
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO bo_experiment_results 
                (experiment_id, iteration, parameters, rating, best_parameters, best_rating)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                result.experiment_id,
                result.iteration,
                json.dumps(result.parameters),
                result.rating,
                json.dumps(result.best_parameters),
                result.best_rating
            ))
            return cursor.lastrowid
        
    def add_pbo_result(self, result: PBOExperimentResult) -> int:
        """Add a new Preferential Bayesian Optimization result to the database.
        
        Args:
            result (PBOExperimentResult): Result from a PBO iteration
            
        Returns:
            int: ID of the inserted result
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO pbo_experiment_results 
                (experiment_id, iteration, parameters_preferred_id, parameters_not_preferred_id, 
                 rating_best, parameters_best)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                result.experiment_id,
                result.iteration,
                result.parameters_preferred_id,
                result.parameters_not_preferred_id,
                result.best_rating,
                json.dumps(result.parameters_best)
            ))
            return cursor.lastrowid
        
    def add_pbo_config(self, config: PBOExperimentConfig) -> int:
        """Add a new Preferential Bayesian Optimization configuration to the database.
        
        Args:
            config (PBOExperimentConfig): Configuration for a PBO iteration
            
        Returns:
            int: ID of the inserted configuration
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO pbo_experiment_configs 
                (experiment_id, parameters_id, parameters, rating)
                VALUES (?, ?, ?, ?)
            """, (
                config.experiment_id,
                config.parameters_id,
                json.dumps(config.parameters),
                config.rating
            ))
            return cursor.lastrowid
        
    def get_prior_type_by_description(self, description: str | None) -> Optional[PriorType]:
        """Retrieve a prior type by description.
        
        Args:
            description (str): The description of the prior type
            
        Returns:
            Optional[PriorType]: The prior type if found, None otherwise
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM prior_types WHERE description = ?", (description if description is not None else "None",))
            row = cursor.fetchone()
            if row:
                return PriorType(
                    id=row[0],
                    description=row[1],
                    created_at=row[2],
                    updated_at=row[3]
                )
        return None

    def get_optimization_type(self, id: int) -> Optional[OptimizationType]:
        """Retrieve an optimization type by ID.
        
        Args:
            id (int): The ID of the optimization type
            
        Returns:
            Optional[OptimizationType]: The optimization type if found, None otherwise
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM optimization_types WHERE id = ?", (id,))
            row = cursor.fetchone()
            if row:
                return OptimizationType(
                    id=row[0],
                    description=row[1],
                    created_at=row[2],
                    updated_at=row[3]
                )
        return None
    
    def get_optimization_type_by_description(self, description: str) -> Optional[OptimizationType]:
        """Retrieve an optimization type by description.
        
        Args:
            description (str): The description of the optimization type
            
        Returns:
            Optional[OptimizationType]: The optimization type if found, None otherwise
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM optimization_types WHERE description = ?", (description,))
            row = cursor.fetchone()
            if row:
                return OptimizationType(
                    id=row[0],
                    description=row[1],
                    created_at=row[2],
                    updated_at=row[3]
                )
        return None

    def get_objective(self, id: int) -> Optional[Objective]:
        """Retrieve an objective by ID.
        
        Args:
            id (int): The ID of the objective
            
        Returns:
            Optional[Objective]: The objective if found, None otherwise
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM objectives WHERE id = ?", (id,))
            row = cursor.fetchone()
            if row:
                return Objective(
                    id=row[0],
                    dimensions=row[1],
                    bounds=json.loads(row[2]),
                    description=row[3],
                    optimizers=json.loads(row[4]),
                    optimal_value=row[5],
                    created_at=row[6],
                    updated_at=row[7]
                )
        return None
    
    def get_objectives_by_description(self, description: str) -> Optional[List[Objective]]:
        """Retrieve an objective by description.
        
        Args:
            description (str): The description of the objective
            
        Returns:
            Optional[List[Objective]]: The objective if found, None otherwise
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM objectives WHERE description = ?", (description,))
            
            objectives = []
            for row in cursor.fetchall():
                objectives.append(Objective(
                    id=row[0],
                    dimensions=row[1],
                    bounds=json.loads(row[2]),
                    description=row[3],
                    optimizers=json.loads(row[4]),
                    optimal_value=row[5],
                    created_at=row[6],
                    updated_at=row[7]
                ))
            
            return objectives
    
    def get_prior_type(self, id: int) -> Optional[PriorType]:
        """Retrieve a prior type by ID.
        
        Args:
            id (int): The ID of the prior type
            
        Returns:
            Optional[PriorType]: The prior type if found, None otherwise
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM prior_types WHERE id = ?", (id,))
            row = cursor.fetchone()
            if row:
                return PriorType(
                    id=row[0],
                    description=row[1],
                    created_at=row[2],
                    updated_at=row[3]
                )
        return None
    
    def get_prior(self, id: int) -> Optional[Prior]:
        """Retrieve a prior by ID.
        
        Args:
            id (int): The ID of the prior
            
        Returns:
            Optional[Prior]: The prior if found, None otherwise
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM priors WHERE id = ?", (id,))
            row = cursor.fetchone()
            if row:
                return Prior(
                    id=row[0],
                    parameters=json.loads(row[1]),
                    prior_type_id=row[2],
                    created_at=row[3],
                    updated_at=row[4]
                )
        return None
    
    def get_priors_by_prior_type_id(self, prior_type_id: int) -> List[Prior]:
        """Retrieve all priors for a specific prior type.
        
        Args:
            prior_type_id (int): ID of the prior type
            
        Returns:
            List[Prior]: List of Prior objects
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM priors WHERE prior_type_id = ?", (prior_type_id,))
            
            priors = []
            for row in cursor.fetchall():
                priors.append(Prior(
                    id=row[0],
                    parameters=json.loads(row[1]),
                    prior_type_id=row[2],
                    created_at=row[3],
                    updated_at=row[4]
                ))
            
            return priors
        
    def get_prior_injection_method(self, id: int) -> Optional[PriorInjectionMethod]:
        """Retrieve a prior injection method by ID.
        
        Args:
            id (int): The ID of the prior injection method
            
        Returns:
            Optional[PriorInjectionMethod]: The prior injection method if found, None otherwise
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM prior_injection_methods WHERE id = ?", (id,))
            row = cursor.fetchone()
            if row:
                return PriorInjectionMethod(
                    id=row[0],
                    description=row[1],
                    created_at=row[2],
                    updated_at=row[3]
                )
        return None
    
    def get_prior_injection_method_by_description(self, description: str) -> Optional[PriorInjectionMethod]:
        """Retrieve a prior injection method by description.
        
        Args:
            description (str): The description of the prior injection method
            
        Returns:
            Optional[PriorInjectionMethod]: The prior injection method if found, None otherwise
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM prior_injection_methods WHERE description = ?", (description,))
            row = cursor.fetchone()
            if row:
                return PriorInjectionMethod(
                    id=row[0],
                    description=row[1],
                    created_at=row[2],
                    updated_at=row[3]
                )
        return None
    
    def get_experiment_config(self, id: int) -> Optional[ExperimentConfig]:
        """Retrieve an experiment configuration by ID.
        
        Args:
            id (int): The ID of the experiment configuration
            
        Returns:
            Optional[ExperimentConfig]: The experiment configuration if found, None otherwise
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiment_configs WHERE id = ?", (id,))
            row = cursor.fetchone()
            if row:
                return ExperimentConfig(
                    id=row[0],
                    optimization_type_id=row[1],
                    objective_id=row[2],
                    prior_id=row[3],
                    prior_injection_method_id=row[4],
                    seed=row[5],
                    num_trials=row[6],
                    num_initial_samples=row[7],
                    num_paths=row[8],
                    created_at=row[9],
                    updated_at=row[10]
                )
        return None
    
    def get_experiment_configs(
            self,
            optimization_type: str | None = None,
            num_trials: int | None = None,
            num_paths: int | None = None,
            num_initial_samples: int | None = None,
            objective_type: str | None = None,
            prior_type: str | None = None,
            seed: int | None = None,
            prior_injection_method: str | None = None,
    ) -> List[ExperimentConfig]:
        """Retrieve experiment configurations based on filtering criteria.
        
        Args:
            optimization_type (str, optional): The optimization type. Defaults to None.
            num_trials (int, optional): The number of trials. Defaults to None.
            num_paths (int, optional): The number of paths. Defaults to None.
            num_initial_samples (int, optional): The number of initial samples. Defaults to None.
            objective_type (str, optional): The objective type. Defaults to None.
            prior_type (str, optional): The prior type. Defaults to None.
            seed (int, optional): The seed. Defaults to None.
            prior_injection_method (str, optional): The prior injection method. Defaults to None.
            
        Returns:
            List[ExperimentConfig]: List of ExperimentConfig objects
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM experiment_configs WHERE 1"
            params = []
            if optimization_type is not None:
                opt_type = self.get_optimization_type_by_description(optimization_type)
                query += " AND optimization_type_id = ?"
                params.append(opt_type.id)
            if num_trials is not None:
                query += " AND num_trials = ?"
                params.append(num_trials)
            if num_paths is not None:
                query += " AND num_paths = ?"
                params.append(num_paths)
            if num_initial_samples is not None:
                query += " AND num_initial_samples = ?"
                params.append(num_initial_samples)
            if objective_type is not None:
                objectives = self.get_objectives_by_description(objective_type)
                objective_ids = [objective.id for objective in objectives]
                query += f" AND objective_id IN ({','.join(['?'] * len(objective_ids))})"
                params.extend(objective_ids)
            if prior_type is not None:
                prior_type_db = self.get_prior_type_by_description(prior_type)
                priors = self.get_priors_by_prior_type_id(prior_type_db.id)
                prior_ids = [prior.id for prior in priors]
                query += f" AND prior_id IN ({','.join(['?'] * len(prior_ids))})"
                params.extend(prior_ids)
            if seed is not None:
                query += " AND seed = ?"
                params.append(seed)
            if prior_injection_method is not None:
                prior_injection_method_db = self.get_prior_injection_method_by_description(prior_injection_method)
                query += " AND prior_injection_method_id = ?"
                params.append(prior_injection_method_db.id)
            
            cursor.execute(query, tuple(params))
            
            configs = []
            for row in cursor.fetchall():
                configs.append(ExperimentConfig(
                    id=row[0],
                    optimization_type_id=row[1],
                    objective_id=row[2],
                    prior_id=row[3],
                    prior_injection_method_id=row[4],
                    seed=row[5],
                    num_trials=row[6],
                    num_initial_samples=row[7],
                    num_paths=row[8],
                    created_at=row[9],
                    updated_at=row[10]
                ))
            
            return configs

    def get_bo_experiment_results(self, experiment_id: int) -> List[BOResult]:
        """Retrieve all results for a specific experiment.
        
        Args:
            experiment_id (int): ID of the experiment
            
        Returns:
            List[BOResult]: List of BOResult objects
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM bo_experiment_results 
                WHERE experiment_id = ? 
                ORDER BY iteration
            """, (experiment_id,))
            
            results = []
            for row in cursor.fetchall():
                results.append(BOResult(
                    id=row[0],
                    experiment_id=row[1],
                    iteration=row[2],
                    parameters=json.loads(row[3]),
                    rating=row[4],
                    best_parameters=json.loads(row[5]),
                    best_rating=row[6],
                    created_at=row[7],
                    updated_at=row[8]
                ))
            
            return results
        
    def get_pbo_experiment_results(self, experiment_id: int) -> List[PBOExperimentResult]:
        """Retrieve all results for a specific PBO experiment.
        
        Args:
            experiment_id (int): ID of the experiment
            
        Returns:
            List[PBOExperimentResult]: List of PBOExperimentResult objects
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM pbo_experiment_results 
                WHERE experiment_id = ? 
                ORDER BY iteration
            """, (experiment_id,))
            
            results = []
            for row in cursor.fetchall():
                results.append(PBOExperimentResult(
                    id=row[0],
                    experiment_id=row[1],
                    iteration=row[2],
                    parameters_preferred_id=row[3],
                    parameters_not_preferred_id=row[4],
                    best_rating=row[5],
                    parameters_best=json.loads(row[6]),
                    created_at=row[7],
                    updated_at=row[8]
                ))
            
            return results
        
    def get_pbo_experiment_configs(self, experiment_id: int) -> List[PBOExperimentConfig]:
        """Retrieve all configurations for a specific PBO experiment.
        
        Args:
            experiment_id (int): ID of the experiment
            
        Returns:
            List[PBOExperimentConfig]: List of PBOExperimentConfig objects
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM pbo_experiment_configs 
                WHERE experiment_id = ? 
                ORDER BY id
            """, (experiment_id,))
            
            configs = []
            for row in cursor.fetchall():
                configs.append(PBOExperimentConfig(
                    id=row[0],
                    experiment_id=row[1],
                    parameters_id=row[2],
                    parameters=json.loads(row[3]),
                    rating=row[4],
                    created_at=row[5],
                    updated_at=row[6]
                ))
            
            return configs
        
    def add_bo_results(self,
        result_X: torch.Tensor,
        result_y: torch.Tensor,
        result_best_X: torch.Tensor,
        result_best_y: torch.Tensor,
        objective: SyntheticTestFunction,
        seed: int,
        n_trials: int,
        n_paths: int,
        n_initial_samples: int,
        temperature: float | None = None,
        prior: UserPriorLocation | None = None,
        prior_type: Literal["Biased", "Unbiased", "BiasedCertain", "UnbiasedUncertain"] | None = None,
        objective_type: str | None = None,
        prior_injection_method: str = "None",
        optimization_type: Literal["BO", "PriorSampling"] = "BO",
    ) -> ExperimentConfig:
        """Add BO results to the database.
        
        Args:
            result_X (torch.Tensor): The X values for the results (shape=(n_trials + n_initial_samples, dimensions))
            result_y (torch.Tensor): The y values for the results (shape=(n_trials + n_initial_samples, 1))
            result_best_X (torch.Tensor): The X values for the best result per iteration (shape=(n_trials, dimensions))
            result_best_y (torch.Tensor): The y values for the best result per iteration (shape=(n_trials,)
            objective (SyntheticTestFunction): The objective function
            seed (int): The random seed used
            n_trials (int): The number of trials
            n_paths (int): The number of paths
            n_initial_samples (int): The number of initial samples
            temperature (float, optional): The temperature for the acquisition function. Defaults to None.
            prior (UserPriorLocation, optional): The user prior. Defaults to None.
            prior_type (Literal["Biased", "Unbiased", "BiasedCertain", "UnbiasedUncertain"], optional): The prior type. Defaults to None.
            objective_type (str, optional): The objective type (to provide a different name than str(objective)). Defaults to None.
            prior_injection_method (str, optional): The prior injection method. Defaults to None.
        """
        # Get optimization type
        opt_type_id = self.get_optimization_type_by_description(optimization_type).id
        
        # Create objective
        obj = Objective(
            dimensions=objective.dim,
            bounds=objective._bounds,
            description=str(objective) if objective_type is None else objective_type,
            optimizers=objective._optimizers,
            optimal_value=objective.optimal_value,
        )
        objective_id = self.add_objective(obj)
        
        # Create prior type
        prior_type_id = self.get_prior_type_by_description(prior_type).id
        
        # Create prior
        parameters = {}
        if temperature is not None:
            parameters["temperature"] = temperature
        prior = Prior(parameters=parameters, prior_type_id=prior_type_id)
        prior_id = self.add_prior(prior)

        # Get prior injection method
        prior_injection_method_id = self.get_prior_injection_method_by_description(prior_injection_method).id
        
        # Create experiment configuration
        config = ExperimentConfig(
            optimization_type_id=opt_type_id,
            objective_id=objective_id,
            prior_id=prior_id,
            prior_injection_method_id=prior_injection_method_id,
            seed=seed,
            num_trials=n_trials,
            num_initial_samples=n_initial_samples,
            num_paths=n_paths
        )
        experiment_id = self.add_experiment_config(config)
        
        # Add results
        for i in range(result_X.shape[0]):
            result = BOResult(
                experiment_id=experiment_id,
                iteration=i + 1,
                parameters=result_X[i].tolist(),
                rating=result_y[i].item(),
                best_parameters=result_best_X[i].tolist(),
                best_rating=result_best_y[i].item()
            )
            self.add_bo_result(result)

        config_from_db = self.get_experiment_config(experiment_id)
        return config_from_db
    
    def add_pbo_results(self,
        result_X: torch.Tensor,
        result_comparisons: torch.Tensor,
        result_best_X: torch.Tensor,
        result_best_y: torch.Tensor,
        result_y: torch.Tensor,
        objective: SyntheticTestFunction,
        seed: int,
        n_trials: int,
        n_paths: int,
        n_initial_samples: int,
        temperature: float | None = None,
        prior: UserPriorLocation | None = None,
        prior_type: Literal["Biased", "Unbiased", "BiasedCertain", "UnbiasedUncertain"] | None = None,
        objective_type: str | None = None,
        prior_injection_method: str = "None",
    ) -> ExperimentConfig:
        """Add PBO results to the database.
        
        Args:
            result_X (torch.Tensor): The X values for the results (shape=(n_trials + n_initial_samples, dimensions))
            result_comparisons (torch.Tensor): The comparisons for the results (shape=(n_trials + n_initial_samples, 2))
            result_best_X (torch.Tensor): The X values for the best result per iteration (shape=(n_trials, dimensions))
            result_y (torch.Tensor): The y values for the results (shape=(n_trials + n_initial_samples, 1))
            result_best_y (torch.Tensor): The y values for the best result per iteration (shape=(n_trials,)
            objective (SyntheticTestFunction): The objective function
            seed (int): The random seed used
            n_trials (int): The number of trials
            n_paths (int): The number of paths
            n_initial_samples (int): The number of initial samples
            temperature (float, optional): The temperature for the acquisition function. Defaults to None.
            prior (UserPriorLocation, optional): The user prior. Defaults to None.
            prior_type (Literal["Biased", "Unbiased", "BiasedCertain", "UnbiasedUncertain"], optional): The prior type. Defaults to None.
            objective_type (str, optional): The objective type (to provide a different name than str(objective)). Defaults to None.
            prior_injection_method (str, optional): The prior injection method. Defaults to 'None'.
        """
        # Get optimization type
        opt_type_id = self.get_optimization_type_by_description("PBO").id

        # Get prior injection method
        prior_injection_method_id = self.get_prior_injection_method_by_description(prior_injection_method).id
        
        # Create objective
        obj = Objective(
            dimensions=objective.dim,
            bounds=objective._bounds,
            description=str(objective) if objective_type is None else objective_type,
            optimizers=objective._optimizers,
            optimal_value=objective.optimal_value,
        )
        objective_id = self.add_objective(obj)
        
        # Create prior type
        prior_type_id = self.get_prior_type_by_description(prior_type).id
        
        # Create prior
        parameters = {}
        if temperature is not None:
            parameters["temperature"] = temperature
        prior = Prior(parameters=parameters, prior_type_id=prior_type_id)
        prior_id = self.add_prior(prior)
        
        # Create experiment configuration
        config = ExperimentConfig(
            optimization_type_id=opt_type_id,
            objective_id=objective_id,
            prior_id=prior_id,
            prior_injection_method_id=prior_injection_method_id,
            seed=seed,
            num_trials=n_trials,
            num_initial_samples=n_initial_samples,
            num_paths=n_paths
        )
        experiment_id = self.add_experiment_config(config)

        # Add parameter entries for each parameter in result_X
        parameters_ids = []
        for i in range(result_X.shape[0]):
            pbo_config = PBOExperimentConfig(
                experiment_id=experiment_id,
                parameters_id=i,
                parameters=result_X[i].tolist(),
                rating=result_y[i].item()
            )
            param_id = self.add_pbo_config(pbo_config)
            parameters_ids.append(param_id)

        # Add results
        for i in range(result_comparisons.shape[0]):
            n_initial_comparisons = result_comparisons.shape[0] - n_trials  # = n_initial_samples * (n_initial_samples - 1) // 2
            result = PBOExperimentResult(
                experiment_id=experiment_id,
                iteration=i + 1,
                parameters_preferred_id=parameters_ids[result_comparisons[i, 0].item()],
                parameters_not_preferred_id=parameters_ids[result_comparisons[i, 1].item()],
                best_rating=result_best_y[i - n_initial_comparisons + 1].item() if i >= n_initial_comparisons else result_y[0].item(),
                parameters_best=result_best_X[i - n_initial_comparisons + 1].tolist() if i >= n_initial_comparisons else result_X[0].tolist()
            )
            self.add_pbo_result(result)

        config_from_db = self.get_experiment_config(experiment_id)
        return config_from_db
    
    def experiment_exists(self, 
        optimization_type: str | None = None,
        num_trials: int | None = None,
        num_paths: int | None = None,
        num_initial_samples: int | None = None,
        objective_type: str | None = None,
        prior_type: str | None = None,
        seed: int | None = None,
        prior_injection_method: str | None = None
    ) -> bool:
        """Check if an experiment configuration exists.
        
        Args:
            optimization_type (str, optional): The optimization type. Defaults to None.
            num_trials (int, optional): The number of trials. Defaults to None.
            num_paths (int, optional): The number of paths. Defaults to None.
            num_initial_samples (int, optional): The number of initial samples. Defaults to None.
            objective_type (str, optional): The objective type. Defaults to None.
            prior_type (str, optional): The prior type. Defaults to None.
            seed (int, optional): The random seed. Defaults to None.
            prior_injection_method (str, optional): The prior injection method. Defaults to None.
            
        Returns:
            bool: True if the experiment exists, False otherwise
        """
        # Get all experiment configurations
        configs = self.get_experiment_configs(
            optimization_type=optimization_type,
            num_trials=num_trials,
            num_paths=num_paths,
            num_initial_samples=num_initial_samples,
            objective_type=objective_type,
            prior_type=prior_type,
            seed=seed,
            prior_injection_method=prior_injection_method
        )

        return len(configs) > 0

def create_database(path: str) -> Database:
    """Create and return a new Database instance."""
    return Database(path)

def get_database(path: str) -> Database:
    """Get an existing Database instance."""
    return Database(path)

def test_database():
    with tempfile.NamedTemporaryFile(suffix='db') as tmp:
        # Create a new database
        db = create_database(tmp.name)
        
        # Add an optimization type
        opt_type = OptimizationType(description="BO")
        opt_type_id = db.add_optimization_type(opt_type)

        # Add a prior injection method
        prior_injection_method = PriorInjectionMethod(description="ColaBO")
        prior_injection_method_id = db.add_prior_injection_method(prior_injection_method)
        
        # Add a prior type
        prior_type = PriorType(description="Unbiased")
        prior_type_id = db.add_prior_type(prior_type)
        
        # Add a prior
        prior = Prior(
            parameters={"m": 5},
            prior_type_id=prior_type_id
        )
        prior_id = db.add_prior(prior)
        
        # Add an objective
        objective = Objective(
            dimensions=2,
            bounds=[(-5, 5), (-5, 5)],
            description='Test objective function',
            optimizers=[(0, 0)],
            optimal_value=0.0,
        )
        objective_id = db.add_objective(objective)

        # Add an optimization type
        opt_type = OptimizationType(description="PBO")
        opt_type_id = db.add_optimization_type(opt_type)
        
        # Add an experiment configuration
        config = ExperimentConfig(
            optimization_type_id=opt_type_id,
            objective_id=objective_id,
            prior_id=prior_id,
            prior_injection_method_id=prior_injection_method_id,
            seed=123,
            num_trials=10,
            num_initial_samples=5,
            num_paths=1
        )
        experiment_id = db.add_experiment_config(config)
        
        # Add some results
        result = BOResult(
            experiment_id=experiment_id,
            iteration=1,
            parameters=[0.5, 0.5],
            rating=0.75,
            best_parameters=[0.5, 0.5],
            best_rating=0.75
        )
        db.add_bo_result(result)
        
        # Retrieve and print results
        results = db.get_bo_experiment_results(experiment_id)
        for result in results:
            print(f"Iteration {result.iteration}: "
                f"Rating = {result.rating}, "
                f"Best Rating = {result.best_rating}")

if __name__ == "__main__":
    test_database()