import streamlit as st

def _get_users():
    try: return dict(st.secrets["users"])
    except Exception: return {}

def authenticate(username: str,password: str):
    record=_get_users().get(username)
    if not record or password != str(record.get("password","")): return None
    return {"username":username,"display_name":str(record.get("display_name",username)),"role":str(record.get("role","user")).lower()}

def is_authenticated(): return bool(st.session_state.get("auth_user"))

def current_user(): return st.session_state.get("auth_user")
def role_is(role: str):
    user=current_user(); return bool(user and user.get("role")==role)
def logout(): st.session_state.auth_user=None
