À)
÷	Æ	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.11.02v2.11.0-rc2-15-g6290819256d8
m
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_name
Variable
f
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:	*
dtype0
q

Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_name
Variable_1
j
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:	*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ô
valueÊBÇ BÀ
-
weights

biases

signatures*
F@
VARIABLE_VALUE
Variable_1"weights/.ATTRIBUTES/VARIABLE_VALUE*
C=
VARIABLE_VALUEVariable!biases/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
z
StaticRegexFullMatchStaticRegexFullMatchsaver_filename"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
\
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
a
Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
h
SelectSelectStaticRegexFullMatchConst_1Const_2"/device:CPU:**
T0*
_output_shapes
: 
`

StringJoin
StringJoinsaver_filenameSelect"/device:CPU:**
N*
_output_shapes
: 
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
Î
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*x
valueoBmB"weights/.ATTRIBUTES/VARIABLE_VALUEB!biases/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
s
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 
Þ
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slicesVariable_1/Read/ReadVariableOpVariable/Read/ReadVariableOpConst"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtypes
2

&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:

MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixessaver_filename"/device:CPU:0*&
 _has_manual_control_dependencies(
i
IdentityIdentitysaver_filename^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
Ñ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*x
valueoBmB"weights/.ATTRIBUTES/VARIABLE_VALUEB!biases/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
v
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 

	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0* 
_output_shapes
:::*
dtypes
2
S

Identity_1Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:

AssignVariableOpAssignVariableOp
Variable_1
Identity_1"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:

AssignVariableOp_1AssignVariableOpVariable
Identity_2"/device:CPU:0*&
 _has_manual_control_dependencies(*
dtype0
E
NoOp_1NoOp"/device:CPU:0*&
 _has_manual_control_dependencies(


Identity_3Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^NoOp_1"/device:CPU:0*
T0*
_output_shapes
:  "
,
saver_filename:0
Identity:0
Identity_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:
G
weights

biases

signatures"
_generic_user_object
:	2Variable
:	2Variable
"
signature_map