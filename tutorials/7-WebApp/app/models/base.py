# -*- coding:utf-8 --*--
from app.extensions import db

# uncomment this when use extra db connection string
# db.bind_db('key')

####################################################################
#
#   base model
#
####################################################################
user_role = db.Table('user_role',
                     db.Column('user_id', db.Integer, db.ForeignKey('user.id')),
                     db.Column('role_id', db.Integer, db.ForeignKey('role.id')))




class User(db.Model):
    """用户"""
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(20))
    name = db.Column(db.String(20))
    # avatar = db.Column(db.String(50))
    password = db.Column(db.String(20))
    cell = db.Column(db.String(20))
    company = db.Column(db.String(100))
    weixin = db.Column(db.String(50))
    openid = db.Column(db.String(50))
    roles = db.relationship('Role', secondary=user_role, lazy='joined')

    def to_dict(self):
        obj = super(User, self).to_dict()
        if len(obj) != 0:
            del obj['password']
        return obj


class Role(db.Model):
    """角色"""
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(40))
    access = db.Column(db.Text)
    # users = db.relationship('User', secondary=user_role, lazy='joined')
    level = db.Column(db.String(64))


