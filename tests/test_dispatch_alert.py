import pytest
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.edge.dispatch_alert import send_sms, send_email, dispatch

def test_send_sms_missing_config():
    # Should return False if config is incomplete
    cfg = {"account_sid": "mock"}
    assert send_sms(cfg, "MINOR", 0.35, 28.0, 77.0) == False

@patch('src.edge.dispatch_alert.Client')
def test_send_sms_success(mock_client):
    cfg = {
        "account_sid": "test_sid",
        "auth_token": "test_token",
        "from_number": "+123",
        "emergency_contact": "+456"
    }
    
    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    
    mock_message = MagicMock()
    mock_message.sid = "SM123"
    mock_instance.messages.create.return_value = mock_message
    
    assert send_sms(cfg, "SEVERE", 0.60, 28.6139, 77.2090) == True
    mock_instance.messages.create.assert_called_once()
    
    # Check if body contains severity and coordinates
    call_kwargs = mock_instance.messages.create.call_args.kwargs
    assert "SEVERE" in call_kwargs['body']
    assert "28.6139" in call_kwargs['body']

def test_send_email_missing_config():
    cfg = {"sender_email": "test@test.com"}
    assert send_email(cfg, "FATAL", 0.90, 28.0, 77.0) == False

@patch('src.edge.dispatch_alert.smtplib.SMTP')
def test_send_email_success(mock_smtp):
    cfg = {
        "sender_email": "sender@test.com",
        "sender_password": "password",
        "hospital_email": "hospital@test.com"
    }
    
    mock_server = MagicMock()
    mock_smtp.return_value = mock_server
    
    assert send_email(cfg, "FATAL", 0.85, 28.0, 77.0) == True
    mock_server.starttls.assert_called_once()
    mock_server.login.assert_called_with("sender@test.com", "password")
    mock_server.send_message.assert_called_once()

@patch('src.edge.dispatch_alert.send_sms')
@patch('src.edge.dispatch_alert.send_email')
@patch('src.edge.dispatch_alert.get_config')
def test_dispatch_severity_mapping(mock_get_config, mock_send_email, mock_send_sms):
    mock_get_config.return_value = {"twilio": {}, "email": {}}
    
    # Test MINOR threshold
    dispatch(0.35, 28.0, 77.0)
    mock_send_sms.assert_called_with({}, "MINOR", 0.35, 28.0, 77.0)
    
    # Test SEVERE threshold
    dispatch(0.50, 28.0, 77.0)
    mock_send_sms.assert_called_with({}, "SEVERE", 0.50, 28.0, 77.0)
    
    # Test FATAL threshold
    dispatch(0.85, 28.0, 77.0)
    mock_send_sms.assert_called_with({}, "FATAL", 0.85, 28.0, 77.0)
