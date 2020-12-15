#-*- coding: utf-8-*-
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


class AuthGroup(db.Model):
    __tablename__ = 'auth_group'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False, unique=True)


class AuthGroupPermission(db.Model):
    __tablename__ = 'auth_group_permissions'
    __table_args__ = (
        db.Index('auth_group_permissions_group_id_permission_id_0cd325b0_uniq', 'group_id', 'permission_id'),
    )

    id = db.Column(db.Integer, primary_key=True)
    group_id = db.Column(db.ForeignKey('auth_group.id'), nullable=False)
    permission_id = db.Column(db.ForeignKey('auth_permission.id'), nullable=False, index=True)

    group = db.relationship('AuthGroup', primaryjoin='AuthGroupPermission.group_id == AuthGroup.id', backref='auth_group_permissions')
    permission = db.relationship('AuthPermission', primaryjoin='AuthGroupPermission.permission_id == AuthPermission.id', backref='auth_group_permissions')


class AuthPermission(db.Model):
    __tablename__ = 'auth_permission'
    __table_args__ = (
        db.Index('auth_permission_content_type_id_codename_01ab375a_uniq', 'content_type_id', 'codename'),
    )

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    content_type_id = db.Column(db.ForeignKey('django_content_type.id'), nullable=False)
    codename = db.Column(db.String(100), nullable=False)

    content_type = db.relationship('DjangoContentType', primaryjoin='AuthPermission.content_type_id == DjangoContentType.id', backref='auth_permissions')


class DetectTargetdetection(db.Model):
    __tablename__ = 'detect_targetdetection'

    id = db.Column(db.Integer, primary_key=True)
    detectType = db.Column(db.Integer)
    xmin = db.Column(db.Integer)
    ymin = db.Column(db.Integer)
    xmax = db.Column(db.Integer)
    ymax = db.Column(db.Integer)
    createAt = db.Column(db.DateTime, nullable=False)
    targetImage_id = db.Column(db.ForeignKey('detect_targetimage.id'), index=True)

    targetImage = db.relationship('DetectTargetimage', primaryjoin='DetectTargetdetection.targetImage_id == DetectTargetimage.id', backref='detect_targetdetections')

    def __init__(self, detectType, xmin, ymin, xmax, ymax, createAt, targetImage_id):
        self.detectType = detectType
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.createAt = createAt
        self.targetImage_id = targetImage_id


class DetectTargetimage(db.Model):
    __tablename__ = 'detect_targetimage'

    id = db.Column(db.Integer, primary_key=True)
    path = db.Column(db.String(254), nullable=False, unique=True)
    createAt = db.Column(db.DateTime, nullable=False)
    target_id = db.Column(db.ForeignKey('hwInfo_product.id'), index=True)

    target = db.relationship('HwInfoProduct', primaryjoin='DetectTargetimage.target_id == HwInfoProduct.id', backref='detect_targetimages')

    def __init__(self, path, createAt, target_id):
        self.path = path
        self.createAt = createAt
        self.target_id = target_id


class DjangoAdminLog(db.Model):
    __tablename__ = 'django_admin_log'

    id = db.Column(db.Integer, primary_key=True)
    action_time = db.Column(db.DateTime, nullable=False)
    object_id = db.Column(db.String)
    object_repr = db.Column(db.String(200), nullable=False)
    action_flag = db.Column(db.SmallInteger, nullable=False)
    change_message = db.Column(db.String, nullable=False)
    content_type_id = db.Column(db.ForeignKey('django_content_type.id'), index=True)
    user_id = db.Column(db.ForeignKey('userApp_user.id'), nullable=False, index=True)

    content_type = db.relationship('DjangoContentType', primaryjoin='DjangoAdminLog.content_type_id == DjangoContentType.id', backref='django_admin_logs')
    user = db.relationship('UserAppUser', primaryjoin='DjangoAdminLog.user_id == UserAppUser.id', backref='django_admin_logs')


class DjangoContentType(db.Model):
    __tablename__ = 'django_content_type'
    __table_args__ = (
        db.Index('django_content_type_app_label_model_76bd3d3b_uniq', 'app_label', 'model'),
    )

    id = db.Column(db.Integer, primary_key=True)
    app_label = db.Column(db.String(100), nullable=False)
    model = db.Column(db.String(100), nullable=False)


class DjangoMigration(db.Model):
    __tablename__ = 'django_migrations'

    id = db.Column(db.Integer, primary_key=True)
    app = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    applied = db.Column(db.DateTime, nullable=False)


class DjangoSession(db.Model):
    __tablename__ = 'django_session'

    session_key = db.Column(db.String(40), primary_key=True)
    session_data = db.Column(db.String, nullable=False)
    expire_date = db.Column(db.DateTime, nullable=False, index=True)


class HwInfoLocation(db.Model):
    __tablename__ = 'hwInfo_location'

    id = db.Column(db.Integer, primary_key=True)
    cordinateX = db.Column(db.Float(asdecimal=True), nullable=False)
    cordinateY = db.Column(db.Float(asdecimal=True), nullable=False)
    updatedAt = db.Column(db.DateTime, nullable=False)
    target_id = db.Column(db.ForeignKey('hwInfo_product.id'), nullable=False, index=True)

    target = db.relationship('HwInfoProduct', primaryjoin='HwInfoLocation.target_id == HwInfoProduct.id', backref='hw_info_locations')


class HwInfoProduct(db.Model):
    __tablename__ = 'hwInfo_product'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    mac = db.Column(db.String(200), nullable=False, unique=True)
    authKey = db.Column(db.String(20))
    isPower = db.Column(db.Integer, nullable=False)
    isActive = db.Column(db.Integer, nullable=False)
    createAt = db.Column(db.DateTime, nullable=False)
    activeAt = db.Column(db.DateTime, nullable=False)
    isPower_date = db.Column(db.DateTime, nullable=False)
    manager_id = db.Column(db.ForeignKey('userApp_user.id'), index=True)

    manager = db.relationship('UserAppUser', primaryjoin='HwInfoProduct.manager_id == UserAppUser.id', backref='hw_info_products')


class RecordStreamrecord(db.Model):
    __tablename__ = 'record_streamrecord'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(254), nullable=False, unique=True)
    createAt = db.Column(db.DateTime, nullable=False)
    target_id = db.Column(db.ForeignKey('hwInfo_product.id'), index=True)

    target = db.relationship('HwInfoProduct', primaryjoin='RecordStreamrecord.target_id == HwInfoProduct.id', backref='record_streamrecords')


class RecordTicrecord(db.Model):
    __tablename__ = 'record_ticrecord'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(254), nullable=False, unique=True)
    createAt = db.Column(db.DateTime, nullable=False)
    target_id = db.Column(db.ForeignKey('hwInfo_product.id'), index=True)

    target = db.relationship('HwInfoProduct', primaryjoin='RecordTicrecord.target_id == HwInfoProduct.id', backref='record_ticrecords')


class StreamInfoStream(db.Model):
    __tablename__ = 'streamInfo_stream'

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(20), nullable=False, unique=True)
    isActive = db.Column(db.Integer, nullable=False)
    startedAt = db.Column(db.DateTime)
    finishedAt = db.Column(db.DateTime)
    target_id = db.Column(db.ForeignKey('hwInfo_product.id'), unique=True)

    target = db.relationship('HwInfoProduct', primaryjoin='StreamInfoStream.target_id == HwInfoProduct.id', backref='stream_info_streams')


class UserAppUser(db.Model):
    __tablename__ = 'userApp_user'

    id = db.Column(db.Integer, primary_key=True)
    password = db.Column(db.String(128), nullable=False)
    last_login = db.Column(db.DateTime)
    first_name = db.Column(db.String(30), nullable=False)
    last_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(254), nullable=False)
    is_staff = db.Column(db.Integer, nullable=False)
    date_joined = db.Column(db.DateTime, nullable=False)
    userid = db.Column(db.String(16), unique=True)
    username = db.Column(db.String(16))
    name = db.Column(db.String(16))
    createdAt = db.Column(db.DateTime, nullable=False)
    activeAt = db.Column(db.DateTime)
    is_active = db.Column(db.Integer, nullable=False)
    is_admin = db.Column(db.Integer, nullable=False)
    is_superuser = db.Column(db.Integer, nullable=False)


class UserAppUserGroup(db.Model):
    __tablename__ = 'userApp_user_groups'
    __table_args__ = (
        db.Index('userApp_user_groups_user_id_group_id_cc6bd041_uniq', 'user_id', 'group_id'),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.ForeignKey('userApp_user.id'), nullable=False)
    group_id = db.Column(db.ForeignKey('auth_group.id'), nullable=False, index=True)

    group = db.relationship('AuthGroup', primaryjoin='UserAppUserGroup.group_id == AuthGroup.id', backref='user_app_user_groups')
    user = db.relationship('UserAppUser', primaryjoin='UserAppUserGroup.user_id == UserAppUser.id', backref='user_app_user_groups')


class UserAppUserUserPermission(db.Model):
    __tablename__ = 'userApp_user_user_permissions'
    __table_args__ = (
        db.Index('userApp_user_user_permis_user_id_permission_id_be62a0c2_uniq', 'user_id', 'permission_id'),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.ForeignKey('userApp_user.id'), nullable=False)
    permission_id = db.Column(db.ForeignKey('auth_permission.id'), nullable=False, index=True)

    permission = db.relationship('AuthPermission', primaryjoin='UserAppUserUserPermission.permission_id == AuthPermission.id', backref='user_app_user_user_permissions')
    user = db.relationship('UserAppUser', primaryjoin='UserAppUserUserPermission.user_id == UserAppUser.id', backref='user_app_user_user_permissions')
