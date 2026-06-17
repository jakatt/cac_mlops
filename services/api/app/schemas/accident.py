"""Pydantic input schema — 28 features expected by the RandomForest model."""
from pydantic import BaseModel, Field


class AccidentFeatures(BaseModel):
    place:             int   = Field(..., description="Place dans le véhicule")
    catu:              int   = Field(..., description="Catégorie d'usager (1=conducteur…)")
    sexe:              int   = Field(..., description="Sexe (1=M, 2=F)")
    secu1:             float = Field(..., description="Équipement de sécurité 1")
    year_acc:          int   = Field(..., description="Année de l'accident")
    victim_age:        float = Field(..., description="Âge de la victime (calculé)")
    catv:              int   = Field(..., description="Catégorie du véhicule (recodé 0-6)")
    obsm:              int   = Field(..., description="Obstacle mobile heurté")
    motor:             int   = Field(..., description="Type de motorisation")
    catr:              int   = Field(..., description="Catégorie de route")
    circ:              float = Field(..., description="Régime de circulation")
    surf:              float = Field(..., description="État de la surface")
    situ:              float = Field(..., description="Situation de l'accident")
    vma:               float = Field(..., description="Vitesse maximale autorisée")
    jour:              int   = Field(..., description="Jour de la semaine")
    mois:              int   = Field(..., description="Mois (1-12)")
    lum:               int   = Field(..., description="Conditions d'éclairage")
    dep:               int   = Field(..., description="Département")
    com:               int   = Field(..., description="Code commune")
    agg_:              int   = Field(..., description="Localisation (1=hors agglo, 2=agglo)")
    intersection_type: int   = Field(..., alias="int", description="Type d'intersection")
    atm:               float = Field(..., description="Conditions atmosphériques (recodé 0/1)")
    col:               float = Field(..., description="Type de collision")
    lat:               float = Field(..., description="Latitude")
    long:              float = Field(..., description="Longitude")
    hour:              int   = Field(..., description="Heure de l'accident")
    nb_victim:         int   = Field(..., description="Nombre de victimes dans l'accident")
    nb_vehicules:      int   = Field(..., description="Nombre de véhicules impliqués")

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    prediction:     int   = Field(..., description="0=non prioritaire, 1=prioritaire")
    probability:    float = Field(..., description="Probabilité de la classe prédite")
    model_version:  str   = Field(..., description="Version du modèle MLflow utilisé")
