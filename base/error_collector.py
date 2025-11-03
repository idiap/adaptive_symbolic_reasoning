# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Lei Xu <lei.xu@idiap.ch>
# SPDX-FileContributor: Pierre Beckmann <pierre.beckmann@idiap.ch>
#
# SPDX-License-Identifier: GPL-3.0-only

import logging
import threading
from typing import List, Dict, Optional
from datetime import datetime

class ErrorCollector(logging.Handler):
    """
    Custom logging handler to collect error information from across the system
    """
    
    def __init__(self, level=logging.ERROR):
        super().__init__(level)
        self.errors = []
        self.lock = threading.RLock()  # Use reentrant lock
        self._current_session_id = None
        
    def set_session_id(self, session_id: str):
        """Set current session ID to identify specific processing runs"""
        try:
            if self.lock.acquire(timeout=1.0):
                try:
                    self._current_session_id = session_id
                finally:
                    self.lock.release()
        except:
            pass
    
    def clear_errors(self):
        """Clear the error list"""
        try:
            if self.lock.acquire(timeout=1.0):
                try:
                    self.errors.clear()
                finally:
                    self.lock.release()
        except:
            pass
    
    def emit(self, record: logging.LogRecord):
        """Handle log records and collect error information"""
        try:
            if record.levelno >= logging.ERROR:
                # Try to acquire lock with timeout to prevent deadlock
                if self.lock.acquire(timeout=1.0):
                    try:
                        message = record.getMessage()
                        
                        # Check for duplicate errors (same message and module in same session)
                        is_duplicate = False
                        error_signature = f"{self._current_session_id}:{record.module}:{message}"
                        
                        # Simple duplicate check: same session + module + message
                        for existing_error in reversed(self.errors[-10:]):  # Check last 10 errors
                            existing_signature = f"{existing_error['session_id']}:{existing_error['module']}:{existing_error['message']}"
                            if existing_signature == error_signature:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            # Extract relative file path for cleaner display
                            import os
                            try:
                                rel_path = os.path.relpath(record.pathname)
                                if rel_path.startswith('..'):
                                    rel_path = record.pathname  # Use full path if relative path goes up
                            except:
                                rel_path = record.pathname
                            
                            error_info = {
                                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                                'level': record.levelname,
                                'message': message,
                                'module': record.module,
                                'function': record.funcName,
                                'line_number': record.lineno,
                                'file_path': rel_path,  # Relative file path
                                'full_pathname': record.pathname,  # Keep full path for reference
                                'location': f"{rel_path}:{record.lineno}",  # Combined location string
                                'session_id': self._current_session_id,
                                'exception_info': None
                            }
                            
                            # If there's exception info, save it as well
                            if record.exc_info:
                                error_info['exception_info'] = self.format_exception(record.exc_info)
                            
                            # Limit the number of errors stored to prevent memory issues
                            if len(self.errors) >= 1000:
                                self.errors = self.errors[-500:]  # Keep only the last 500 errors
                            
                            self.errors.append(error_info)
                    finally:
                        self.lock.release()
        except Exception:
            # If error collection fails, don't break the main program
            pass
    
    def format_exception(self, exc_info):
        """Format exception information"""
        import traceback
        return ''.join(traceback.format_exception(*exc_info))
    
    def get_errors(self, session_id: Optional[str] = None) -> List[Dict]:
        """Get error list, optionally filtered by session ID"""
        try:
            if self.lock.acquire(timeout=1.0):
                try:
                    if session_id is None:
                        return self.errors.copy()
                    else:
                        return [error for error in self.errors if error['session_id'] == session_id]
                finally:
                    self.lock.release()
        except:
            # If lock fails, return empty list to prevent program hanging
            return []
    
    def get_error_summary(self, session_id: Optional[str] = None) -> Dict:
        """Get simplified error summary as list of dictionaries with error_info and location"""
        errors = self.get_errors(session_id)
        
        # Create list of dictionaries, each containing error_info and location
        error_list = []
        
        for error in errors:
            error_dict = {
                'error_info': error['message'],
                'location': error.get('location', f"{error.get('file_path', 'unknown')}:{error.get('line_number', 0)}")
            }
            error_list.append(error_dict)
        
        summary = {
            'total_errors': len(errors),
            'errors': error_list
        }
        
        return summary

# Global error collector instance
_global_error_collector = None

def get_error_collector() -> ErrorCollector:
    """Get global error collector instance"""
    global _global_error_collector
    if _global_error_collector is None:
        _global_error_collector = ErrorCollector()
    return _global_error_collector

def setup_error_collection():
    """Setup error collection by adding error collector to root logger"""
    collector = get_error_collector()
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Check if error collector is already added
    for handler in root_logger.handlers:
        if isinstance(handler, ErrorCollector):
            return collector
    
    # Add error collector to root logger
    root_logger.addHandler(collector)
    
    # Also add to the agentic_sdk logger specifically
    agentic_logger = logging.getLogger("agentic_sdk")
    if collector not in agentic_logger.handlers:
        agentic_logger.addHandler(collector)
    
    # Set level to ensure ERROR messages are captured
    root_logger.setLevel(logging.DEBUG)
    agentic_logger.setLevel(logging.DEBUG)
    
    return collector
