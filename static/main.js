$("#inputTag").tagsinput('items');

$(document).ready(function() {
$('#sendbutton').on('submit', function(event) {
  console.log(1)

  // $.ajax({
  //   type: 'POST',
  //   url: "/proc",
  //   data: JSON.stringify("hello"),
  //   contentType: "application/json;charset=utf-8",
  //   dataType: "json",
  //   success: function(data){
  //     // do something with the received data
  //     console.log(data)
  //   }
  // })
})
});
