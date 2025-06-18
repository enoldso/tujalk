"""
Utility functions for role-based access control and department management
"""

from functools import wraps
from flask import abort, redirect, url_for, request
from flask_login import current_user

# Department permissions mapping
DEPARTMENT_PERMISSIONS = {
    'super_admin': ['all'],  # Can access everything
    'administration': ['users', 'departments', 'reports', 'settings'],
    'clinical': ['patients', 'appointments', 'messages', 'health_tips', 'prescriptions', 'walk_in'],
    'finance': ['billing', 'payments', 'financial_reports'],
    'laboratory': ['lab_tests', 'lab_results', 'lab_reports'],
    'pharmacy': ['prescriptions', 'inventory', 'dispensing'],
    'reception': ['patients', 'appointments', 'walk_in', 'basic_billing']
}

# Route to department mapping
ROUTE_DEPARTMENTS = {
    'finance': 'finance',
    'billing': 'finance',
    'payments': 'finance', 
    'create_bill': 'finance',
    'record_payment': 'finance',
    'update_bill_status': 'finance',
    'update_payment_status': 'finance',
    
    'lab_tests': 'laboratory',
    'order_lab_test': 'laboratory',
    'lab_test_results': 'laboratory',
    
    'patients': 'clinical',
    'patient_detail': 'clinical',
    'appointments': 'clinical',
    'messages': 'clinical',
    'patient_messages': 'clinical',
    'health_tips': 'clinical',
    'health_education': 'clinical',
    'prescriptions': 'clinical',
    'create_prescription': 'clinical',
    'walk_in_patients': 'clinical',
    'register_walk_in': 'clinical',
    
    'admin_dashboard': 'administration',
    'manage_users': 'administration',
    'manage_departments': 'administration',
    'system_reports': 'administration'
}

def requires_permission(permission):
    """Decorator to check if user has required permission"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for('login'))
            
            # Super admin has access to everything
            if hasattr(current_user, 'role') and current_user.role == 'super_admin':
                return f(*args, **kwargs)
            
            # Check if user has the required permission
            if hasattr(current_user, 'has_permission') and current_user.has_permission(permission):
                return f(*args, **kwargs)
            
            # Check department access
            if hasattr(current_user, 'can_access_department'):
                required_dept = ROUTE_DEPARTMENTS.get(request.endpoint)
                if required_dept and current_user.can_access_department(required_dept):
                    return f(*args, **kwargs)
            
            abort(403)  # Forbidden
        return decorated_function
    return decorator

def requires_department(department):
    """Decorator to check if user belongs to required department"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for('login'))
            
            # Super admin has access to everything
            if hasattr(current_user, 'role') and current_user.role == 'super_admin':
                return f(*args, **kwargs)
            
            # Check department access
            if hasattr(current_user, 'can_access_department') and current_user.can_access_department(department):
                return f(*args, **kwargs)
            
            abort(403)  # Forbidden
        return decorated_function
    return decorator

def get_user_accessible_routes():
    """Get list of routes the current user can access"""
    if not current_user.is_authenticated:
        return []
    
    # Super admin can access everything
    if hasattr(current_user, 'role') and current_user.role == 'super_admin':
        return list(ROUTE_DEPARTMENTS.keys())
    
    accessible_routes = []
    user_department = getattr(current_user, 'department', None)
    user_permissions = getattr(current_user, 'permissions', [])
    
    for route, required_dept in ROUTE_DEPARTMENTS.items():
        if user_department == required_dept or required_dept in user_permissions:
            accessible_routes.append(route)
    
    return accessible_routes

def get_navigation_items():
    """Get navigation items based on user's role and department"""
    if not current_user.is_authenticated:
        return []
    
    accessible_routes = get_user_accessible_routes()
    
    # Define navigation structure
    nav_items = []
    
    # Dashboard (always visible for authenticated users)
    if hasattr(current_user, 'role'):
        if current_user.role == 'super_admin':
            nav_items.append({'name': 'Admin Dashboard', 'route': 'admin_dashboard', 'icon': 'settings'})
        elif current_user.department == 'finance':
            nav_items.append({'name': 'Finance Dashboard', 'route': 'finance_dashboard', 'icon': 'dollar-sign'})
        elif current_user.department == 'laboratory':
            nav_items.append({'name': 'Lab Dashboard', 'route': 'lab_dashboard', 'icon': 'search'})
        else:
            nav_items.append({'name': 'Dashboard', 'route': 'dashboard', 'icon': 'home'})
    
    # Clinical section
    if any(route in accessible_routes for route in ['patients', 'appointments', 'prescriptions']):
        nav_items.extend([
            {'name': 'Patients', 'route': 'patients', 'icon': 'users'},
            {'name': 'Appointments', 'route': 'appointments', 'icon': 'calendar'},
            {'name': 'Messages', 'route': 'messages', 'icon': 'mail'},
            {'name': 'Prescriptions', 'route': 'prescriptions', 'icon': 'file-text'},
            {'name': 'Walk-In', 'route': 'walk_in_patients', 'icon': 'user-plus'},
        ])
    
    # Finance section
    if any(route in accessible_routes for route in ['finance', 'billing', 'payments']):
        nav_items.append({'name': 'Finance', 'route': 'finance', 'icon': 'dollar-sign'})
    
    # Laboratory section
    if any(route in accessible_routes for route in ['lab_tests']):
        nav_items.append({'name': 'Lab Tests', 'route': 'lab_tests', 'icon': 'search'})
    
    # Admin section
    if current_user.role == 'super_admin':
        nav_items.extend([
            {'name': 'User Management', 'route': 'manage_users', 'icon': 'user-check'},
            {'name': 'Departments', 'route': 'manage_departments', 'icon': 'layers'},
        ])
    
    return nav_items

def can_access_route(route_name):
    """Check if current user can access a specific route"""
    if not current_user.is_authenticated:
        return False
    
    accessible_routes = get_user_accessible_routes()
    return route_name in accessible_routes