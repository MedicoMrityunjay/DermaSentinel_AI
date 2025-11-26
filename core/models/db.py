from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from core.database import Base

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    scans = relationship("Scan", back_populates="patient")

class Scan(Base):
    __tablename__ = "scans"

    id = Column(String, primary_key=True, index=True) # UUID
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    
    diagnosis = Column(String)
    probability = Column(Float)
    severity = Column(String)
    uncertainty = Column(Float)
    
    age = Column(Integer)
    sex = Column(String)
    site = Column(String)
    
    mask_path = Column(String, nullable=True) # Path to saved mask image if needed
    metadata_json = Column(Text) # JSON string for concepts, clinical notes, etc.
    
    timestamp = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="scans")
